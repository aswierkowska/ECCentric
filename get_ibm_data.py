from qiskit_ibm_runtime import QiskitRuntimeService

service = QiskitRuntimeService(
    channel='ibm_quantum',
    instance='crn:v1:bluemix:public:quantum-computing:us-east:a/05111ea2e52d4ef38773605493c21b73:98e15c53-aab4-4ec2-8247-3b88cd739a95::'
)
job = service.job('d55aqj3ht8fs73a0mscg')
job_result = job.result()

for idx, pub_result in enumerate(job_result):
    print(f"Expectation values for pub {idx}: {pub_result.data.evs}")