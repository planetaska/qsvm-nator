import os
from dotenv import load_dotenv
from qiskit_ibm_runtime import QiskitRuntimeService

load_dotenv()
RUNTIME_TOKEN = os.getenv("IBM_RUNTIME_TOKEN")

# Save an IBM Quantum account and set it as your default account.
# You will run this script only once to setup your quantum runtime service.
QiskitRuntimeService.save_account(
    channel="ibm_quantum",
    token=RUNTIME_TOKEN,
    set_as_default=True,
    # Use `overwrite=True` if you're updating your token.
    overwrite=True,
)
