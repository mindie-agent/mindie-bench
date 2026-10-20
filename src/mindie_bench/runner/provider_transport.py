"""Provider connection planning using the parent's enterprise network settings.

[Role] Construct the gateway's single HTTP connection and request target.
[Invariants] HTTP CONNECT retains verified origin TLS; proxy credentials stay at
the proxy. Numeric-loopback HTTP relays and NO_PROXY connect directly. Unsupported
proxy protocols never fall back to direct.
[Timeouts & Locks] The gateway owns connection tracking and its 120-second timeout.
[Side-Effect Policy] No requests or retries here; the gateway submits and meters.
"""

import base64
import http.client
from urllib.parse import unquote, urlsplit
from urllib.request import getproxies, proxy_bypass


def provider_connection(provider_url, request_path):
    endpoint = urlsplit(provider_url)
    target = endpoint.path.rstrip("/") + "/" + request_path.removeprefix("/v1/").lstrip("/")
    connection_type = (
        http.client.HTTPSConnection if endpoint.scheme == "https" else http.client.HTTPConnection
    )
    proxies = getproxies()
    proxy_url = proxies.get(endpoint.scheme) or proxies.get("all")
    # Gateway admits cleartext providers only for numeric-loopback local relays.
    if endpoint.scheme == "http" or not proxy_url or proxy_bypass(endpoint.netloc):
        return connection_type(endpoint.hostname, endpoint.port, timeout=120), target

    proxy = urlsplit(proxy_url if "://" in proxy_url else "http://" + proxy_url)
    if proxy.scheme != "http":
        raise ValueError("model gateway requires an http:// CONNECT proxy")
    headers = {}
    if proxy.username is not None:
        credentials = unquote(proxy.username) + ":" + unquote(proxy.password or "")
        headers["Proxy-Authorization"] = "Basic " + base64.b64encode(credentials.encode()).decode()
    connection = connection_type(proxy.hostname, proxy.port or 80, timeout=120)
    # HTTPSConnection performs CONNECT first, then verifies the origin's TLS.
    connection.set_tunnel(endpoint.hostname, endpoint.port or 443, headers=headers)
    return connection, target
