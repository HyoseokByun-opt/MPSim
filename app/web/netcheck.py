"""Why can other machines not reach this server?

Almost never the application: it binds 0.0.0.0 and the operating system either
lets the packet in or does not. This module reports the four things that
actually decide it, so the answer takes a minute instead of an afternoon:

  1. which addresses this machine has, and which one to hand out
  2. whether anything is already holding the port
  3. whether Windows Firewall has an inbound rule for *this* Python
  4. what is left when those are all fine — which is a network policy problem
     (wireless client isolation, or a different subnet), not a server problem

Nothing here changes the system. The firewall command is printed for a person
to run knowingly, with administrator rights, because opening a port on a
machine is not a decision a program should take on someone's behalf.
"""
import os
import socket
import subprocess
import sys


def _ps(script):
    """Run PowerShell and return stdout, or '' if it is unavailable."""
    try:
        r = subprocess.run(
            ["powershell", "-NoProfile", "-NonInteractive", "-Command", script],
            capture_output=True, text=True, timeout=25)
        return r.stdout.strip()
    except Exception:                                        # noqa: BLE001
        return ""


def local_addresses():
    """Every IPv4 address this machine answers on, best guess first."""
    addrs = []
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        addrs.append(s.getsockname()[0])
        s.close()
    except Exception:                                        # noqa: BLE001
        pass
    try:
        for info in socket.getaddrinfo(socket.gethostname(), None,
                                       socket.AF_INET):
            ip = info[4][0]
            if ip not in addrs and not ip.startswith("127."):
                addrs.append(ip)
    except Exception:                                        # noqa: BLE001
        pass
    return addrs


def port_in_use(port, host="0.0.0.0"):
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    try:
        s.bind((host, port))
        return False
    except OSError:
        return True
    finally:
        s.close()


def firewall_state():
    """-> dict describing Windows Firewall as it applies to this Python."""
    if not sys.platform.startswith("win"):
        return {"platform": "not windows"}
    exe = os.path.abspath(sys.executable).lower()
    profile = _ps("(Get-NetConnectionProfile | Select-Object -First 1)"
                  ".NetworkCategory")
    enabled = _ps("(Get-NetFirewallProfile | Where-Object Enabled -eq True |"
                  " Select-Object -ExpandProperty Name) -join ','")
    # Ask the application filters for python first and only then resolve
    # those few rules. Walking every inbound rule and calling
    # Get-NetFirewallApplicationFilter on each takes long enough to hit the
    # timeout on a managed machine, and a timeout that looks like "no rule"
    # would send someone off to run an administrator command they do not need.
    rules = _ps(
        "Get-NetFirewallApplicationFilter -All |"
        " Where-Object { $_.Program -like '*python*' } |"
        " ForEach-Object { $p = $_.Program; $r = $_ | Get-NetFirewallRule"
        " -ErrorAction SilentlyContinue;"
        " if ($r -and $r.Enabled -eq 'True' -and $r.Direction -eq 'Inbound')"
        " { \"$($r.Action)|$($r.Profile)|$p\" } }")
    known = bool(profile or enabled or rules)

    def pick(action):
        out = []
        for ln in rules.splitlines():
            parts = ln.split("|")
            if len(parts) < 3:
                continue
            if parts[0].strip().lower() != action:
                continue
            if parts[-1].strip().lower() == exe:
                out.append(parts[1].strip())
        return out

    return {"platform": "windows", "python": exe, "profile": profile,
            "enabled_profiles": enabled, "known": known,
            "allow_rules": pick("allow"), "block_rules": pick("block")}


HOSTS = r"C:\Windows\System32\drivers\etc\hosts"


def resolves(name):
    """The address this name resolves to right now, or None."""
    try:
        return socket.gethostbyname(name)
    except Exception:                                        # noqa: BLE001
        return None


def computer_name():
    """This machine's own name.

    Windows PCs on the same subnet can usually reach each other by name with
    no DNS record and no hosts file, which makes this the one option that
    needs nothing set up anywhere.
    """
    try:
        return socket.gethostname()
    except Exception:                                        # noqa: BLE001
        return None


def hosts_line(name, ip):
    return "%-15s %s" % (ip, name)


def _add_hosts_script(name, ip):
    """An elevated one-liner that adds the mapping only if it is missing."""
    pattern = name.replace(".", "\\.")
    return ("if (-not (Select-String -Path '{h}' -Pattern '{p}' -Quiet)) "
            "{{ Add-Content -Path '{h}' -Value '{line}' -Encoding ASCII }}; "
            "ipconfig /flushdns | Out-Null"
            ).format(h=HOSTS, p=pattern, line=hosts_line(name, ip))


def add_hosts(name, ip, out=print):
    """Ask Windows for elevation and map `name` to `ip` on this PC.

    Editing the hosts file changes how every program on the machine resolves
    that name, so it never happens as a side effect of starting the server. It
    happens only when someone asks for it by name, and Windows puts its own
    consent prompt in front of it.
    """
    if not sys.platform.startswith("win"):
        out("  this helper is Windows-only; add this line to /etc/hosts:")
        out("      %s" % hosts_line(name, ip))
        return False
    already = resolves(name)
    if already:
        out("  %s already resolves to %s -- nothing to do" % (name, already))
        return True
    out("  adding to %s" % HOSTS)
    out("      %s" % hosts_line(name, ip))
    out("  Windows will now ask for administrator permission ...")
    inner = _add_hosts_script(name, ip).replace("'", "''")
    try:
        subprocess.run(
            ["powershell", "-NoProfile", "-Command",
             "Start-Process powershell -Verb RunAs -Wait -WindowStyle Hidden "
             "-ArgumentList '-NoProfile','-Command','%s'" % inner],
            capture_output=True, text=True, timeout=180)
    except Exception as e:                                   # noqa: BLE001
        out("  could not launch the elevated step: %s" % e)
        return False
    got = resolves(name)
    if got:
        out("  done -- %s now resolves to %s" % (name, got))
        return True
    out("  it still does not resolve: either the prompt was declined, or")
    out("  something else already owns that name. By hand: open Notepad as")
    out("  administrator, open %s and add:" % HOSTS)
    out("      %s" % hosts_line(name, ip))
    return False


def icmp_allowed():
    """Is inbound ping enabled?  -> True / False / None if unknown.

    Windows disables inbound ICMP echo by default, so a failed ping proves
    nothing about whether a TCP port is reachable. Reporting "ping fails, so
    the network is broken" sends people to argue with their network team about
    a setting on their own machine.
    """
    if not sys.platform.startswith("win"):
        return None
    out = _ps("(Get-NetFirewallRule -Name '*ICMP4-ERQ*' -ErrorAction "
              "SilentlyContinue | Where-Object { $_.Enabled -eq 'True' -and "
              "$_.Direction -eq 'Inbound' } | Measure-Object).Count")
    try:
        return int(out.strip()) > 0
    except Exception:                                        # noqa: BLE001
        return None


def icmp_command():
    return ("Set-NetFirewallRule -Name '*ICMP4-ERQ*' -Enabled True   "
            "# only needed to make ping work for diagnosis")


def subnets():
    """(address, prefix, adapter) for every real IPv4 interface."""
    if not sys.platform.startswith("win"):
        return []
    out = _ps("Get-NetIPAddress -AddressFamily IPv4 | Where-Object "
              "{ $_.IPAddress -notlike '127.*' } | ForEach-Object "
              "{ \"$($_.IPAddress)|$($_.PrefixLength)|$($_.InterfaceAlias)\" }")
    rows = []
    for ln in out.splitlines():
        parts = ln.split("|")
        if len(parts) == 3:
            rows.append(tuple(p.strip() for p in parts))
    return rows


def neighbours():
    """Who this machine can actually see on its own network segment.

    -> (peers, gateway) where peers excludes the gateway and broadcast.

    This is the test that separates the two causes people confuse. A wireless
    access point doing client isolation lets every station reach the gateway
    and nothing else, so the neighbour table contains the gateway alone. No
    firewall setting on either machine can change that, which is why finding
    it early saves a lot of pointless firewall work.
    """
    if not sys.platform.startswith("win"):
        return None, None
    gw = _ps("(Get-NetRoute -DestinationPrefix '0.0.0.0/0' -ErrorAction "
             "SilentlyContinue | Select-Object -First 1).NextHop").strip()
    raw = _ps("Get-NetNeighbor -AddressFamily IPv4 -ErrorAction "
              "SilentlyContinue | Where-Object { $_.State -ne 'Unreachable' "
              "-and $_.State -ne 'Permanent' } | ForEach-Object "
              "{ $_.IPAddress }")
    seen = [ln.strip() for ln in raw.splitlines() if ln.strip()]
    peers = [ip for ip in seen
             if ip != gw and not ip.endswith(".255")
             and not ip.startswith("169.254.") and not ip.startswith("224.")
             and not ip.startswith("239.")]
    return peers, (gw or None)


def firewall_command(port, name="MPSim"):
    return ("New-NetFirewallRule -DisplayName '%s' -Direction Inbound "
            "-Action Allow -Protocol TCP -LocalPort %d "
            "-Profile Domain,Private,Public" % (name, port))


def report(port, host, server_name=None, out=print):
    """Print the whole picture.  Returns True if nothing obvious is wrong."""
    ok = True
    addrs = local_addresses()
    out("network check")
    out("-" * 66)

    out("  addresses on this machine : %s" % (", ".join(addrs) or "none found"))
    if host not in ("0.0.0.0", "::"):
        out("  BOUND TO %s ONLY -- other machines cannot reach that." % host)
        out("      start it with --host 0.0.0.0 to serve the network")
        ok = False
    else:
        out("  binding                   : 0.0.0.0 (every interface)")

    if port_in_use(port):
        out("  port %-5d                : ALREADY IN USE by another process"
            % port)
        out("      pick another with --port, or stop what is holding it")
        ok = False
    else:
        out("  port %-5d                : free" % port)

    fw = firewall_state()
    if fw.get("platform") == "windows":
        out("  network profile           : %s" % (fw["profile"] or "unknown"))
        out("  firewall enabled for      : %s" % (fw["enabled_profiles"] or "-"))
        if not fw.get("known"):
            out("  inbound rule for Python   : could not be determined "
                "(PowerShell did not answer)")
            out("      check by hand, or just add the rule; it is harmless "
                "if one already exists:")
            out("      %s" % firewall_command(port))
        elif fw["block_rules"]:
            out("  BLOCK RULE for this Python: profile %s"
                % fw["block_rules"][0])
            out("      a Block rule beats an Allow rule; remove it")
            ok = False
        elif fw["allow_rules"]:
            profiles = ", ".join(sorted(set(fw["allow_rules"])))
            out("  inbound rule for Python   : present, profile %s" % profiles)
            out("      a rule for the port itself is more robust than one tied")
            out("      to a program path; in an ADMINISTRATOR PowerShell:")
            out("      %s" % firewall_command(port))
            cur = (fw["profile"] or "").strip()
            if cur and not any(cur.lower() in p.lower() or p.lower() == "any"
                               for p in fw["allow_rules"]):
                out("      but this network is '%s' and the rule does not "
                    "cover it" % cur)
                out("      %s" % firewall_command(port))
                ok = False
        else:
            out("  inbound rule for Python   : NONE -- Windows will drop "
                "connections from other machines")
            out("      in an ADMINISTRATOR PowerShell (not cmd), run:")
            out("      %s" % firewall_command(port))
            ok = False

    peers, gw = neighbours()
    if peers is not None:
        if gw and not peers:
            out("  visible on this segment   : ONLY the gateway (%s)" % gw)
            out("      This is the signature of wireless client isolation:")
            out("      every device reaches the internet and none reach each")
            out("      other. No firewall setting on either PC can change it.")
            out("      -> use a wired connection, or ask the network team to")
            out("         allow peer traffic on this SSID/VLAN.")
            out("      -> to confirm in 30 seconds: put both PCs on a phone")
            out("         hotspot and try again. If it works there, this is it.")
            ok = False
        elif peers:
            out("  visible on this segment   : gateway %s, plus %d other "
                "device(s)" % (gw or "?", len(peers)))
            out("      %s" % ", ".join(peers[:6]))

    icmp = icmp_allowed()
    if icmp is False:
        out("  inbound ping (ICMP)       : DISABLED (the Windows default)")
        out("      so a failed ping means nothing here -- do not use it as")
        out("      the test. To make ping usable, in an ADMIN PowerShell:")
        out("      %s" % icmp_command())
    elif icmp:
        out("  inbound ping (ICMP)       : allowed")

    nets = subnets()
    if nets:
        out("  interfaces                :")
        for ip_, plen, alias in nets:
            out("      %-16s /%-3s %s" % (ip_, plen, alias))

    tip = addrs[0] if addrs else "<this-ip>"
    url = "http://%s%s" % (tip, "" if port == 80 else ":%d" % port)
    out("")
    out("  TEST FROM THE OTHER PC -- test the port, not ping.")
    out("  The simplest test needs no command line at all: open a browser")
    out("  on that PC and go to")
    out("      %s" % url)
    out("")
    out("  In Command Prompt (cmd) -- curl ships with Windows 10 and 11:")
    out("      curl -m 5 -v %s/healthz" % url)
    out("      ipconfig | findstr IPv4")
    out("")
    out("  In PowerShell (blue window; Test-NetConnection is a PowerShell")
    out("  command and does not exist in cmd):")
    out("      Test-NetConnection %s -Port %d" % (tip, port))
    out("")
    out("  the browser loads, or")
    out("  curl prints HTTP/1.1   -> it works")
    out("  its address is NOT")
    out("  192.168.200.x-like      -> you are on different subnets. Nothing on")
    out("                             this machine can fix that; both PCs need")
    out("                             the same network, or a route between")
    out("  same subnet, but")
    out("  the port fails          -> either no inbound rule for the port (add")
    out("                             the one printed above), or the access")
    out("                             point is isolating its clients. Corporate")
    out("                             and guest Wi-Fi very often do: every")
    out("                             device reaches the internet and none")
    out("                             reach each other. A wired connection, or")
    out("                             both PCs on the wired VLAN, is the fix.")
    out("                             A phone hotspot with both PCs on it is a")
    out("                             30-second way to prove that is the cause.")
    ip = addrs[0] if addrs else "<this-ip>"
    cn = computer_name()
    if cn and resolves(cn):
        out("")
        out("  no-setup option: this machine is called '%s', and Windows PCs"
            % cn)
        out("  on the same subnet can usually reach it by that name with no")
        out("  DNS and no hosts file:   http://%s%s"
            % (cn, "" if port == 80 else ":%d" % port))
    if server_name:
        got = resolves(server_name)
        out("")
        if got:
            out("  %s resolves to %s" % (server_name, got))
            if got not in addrs:
                out("      but that is NOT this machine -- the name points "
                    "somewhere else")
                ok = False
        else:
            out("  %s DOES NOT RESOLVE anywhere yet, including on this"
                % server_name)
            out("  machine, so that link cannot open. Pick one:")
            out("    - run:  python run_web.py --add-hosts --server-name %s"
                % server_name)
            out("      (this PC only; repeat on each client PC, needs admin)")
            out("    - or ask IT for an internal DNS A record:")
            out("          %s  ->  %s" % (server_name, ip))
            out("      that is the only option that scales past a few people")
            ok = False
    out("-" * 66)
    return ok
