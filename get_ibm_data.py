from qiskit_ibm_runtime import QiskitRuntimeService

service = QiskitRuntimeService(
    channel='ibm_quantum',
    instance='crn:v1:bluemix:public:quantum-computing:us-east:a/05111ea2e52d4ef38773605493c21b73:98e15c53-aab4-4ec2-8247-3b88cd739a95::'
)
job = service.job('d594idfp3tbc73arpkhg')
job_result = job.result()

pub_result = job_result[0].data.block_0_meas_block_0.get_counts()
print(pub_result)

#for idx, pub_result in enumerate(job_result):
#    print(f"Expectation values for pub {idx}: {pub_result.data.evs}")


counts = pub_result  # your dict

only_00000000 = counts.get("00000000", 0)
everything_else = sum(v for k, v in counts.items() if k != "00000000")
total = only_00000000 + everything_else  # or sum(counts.values())

print("only_00000000:", only_00000000)
print("everything_else:", everything_else)
print("total:", total)

print("0/total: " + str(only_00000000 / total))
print("logical error rate: " + str(1 - only_00000000 / total))
