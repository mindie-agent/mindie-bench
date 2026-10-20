# Local TLS test fixture

`test-cert.pem` / `test-key.pem` are a self-signed localhost certificate and key
used only by `tests/test_gateway_tls_observations.py` to exercise certificate
validation and a peer fatal alert on a loopback TLS endpoint.

They are not trusted by system or production trust stores, and the key has no
operational use outside this test.
