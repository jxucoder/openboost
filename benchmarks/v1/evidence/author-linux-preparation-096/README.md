# Linux author-worker local preparation

At clean `cda1947b9759118caf3c7a0752db070fc75be9b7`, the local input check and
installed Modal SDK image construction pass. [Manifest](manifest.json) records
the source revision, actual host/Python, freeze hash and raw artifact hashes.
The [exact CLI result](check-command.json) verifies 22 frozen files and an explicit
13-file proposed upload containing six executable public-document cases plus
thirteen isolation/integrity observations. [SDK inspection](sdk.json) records
`modal==1.3.0.post1`, actual signatures and construction of the image definition.

No image was built remotely, no file was uploaded and no Sandbox or model ran.
These are local preparation results. Remote isolation is `not_run`, generated
tokens remain unknown and independent dispatch remains false. The classifier's
synthetic unit inputs are not archived as measured provider outcomes.

The [pending CPU smoke freeze](../../../../v1-sprints/096-linux-worker-smoke.json)
names every upload and source digest, one 90-second Sandbox, 2 CPUs, 2048 MiB,
zero GPU and zero application retries. A required image build is separate; its
output is captured in `build.log` during execution. The next actual run must retain
all failures and stop for reflection. [Sprint 096](../../../../v1-sprints/096-linux-author-worker.md)
defines acceptance and the remaining token/arm/settings gates. The independent
GPU run-8 freeze retains all 85 original source hashes.
