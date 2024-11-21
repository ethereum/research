import quadratic_provers as q

data = q.eval_across_field([1, 2, 3, 4], 11)
qproof = q.mk_quadratic_proof(data, 4, 11)
assert q.check_quadratic_proof(data, qproof, 4, 5, 11)
data2 = q.eval_across_field(range(36), 97)
cproof = q.mk_column_proof(data2, 36, 97)
assert q.check_column_proof(data2, cproof, 36, 10, 97)
