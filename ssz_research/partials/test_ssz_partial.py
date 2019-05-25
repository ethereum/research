from minimal_ssz import SSZType, infer_type, Vector, hash_tree_root
from ssz_partial import ssz_branch, merge_ssz_branches, SSZPartial, descend
from ssz_partial import get_generalized_indices, get_proof_indices
import os, random

Person = SSZType({"is_male": "bool", "age": "uint64", "name": "bytes"})
City = SSZType({"coords": ["uint64", 2], "people": [Person]})

people = [
    Person(is_male=True, age=45, name=b"Alex"),
    Person(is_male=True, age=47, name=b"Bob"),
    Person(is_male=True, age=49, name=b"Carl"),
    Person(is_male=True, age=51, name=b"Danny"),
    Person(is_male=True, age=53, name=b"Evan"),
    Person(is_male=False, age=55, name=b"Fae"),
    Person(is_male=False, age=57, name=b"Ginny"),
    Person(is_male=False, age=59, name=b"Heather"),
    Person(is_male=False, age=61, name=b"Ingrid"),
    Person(is_male=False, age=63, name=b"Kane"),
]

city = City(coords=Vector([45, 90]), people=people)

paths = [
    ["coords", 0],
    ["people", 4, "name", 1],
    ["people", 9, "is_male"],
    ["people", 7],
    ["people", 1],
]

p = SSZPartial(infer_type(city), merge_ssz_branches(*[ssz_branch(city, path) for path in paths]))
object_keys = sorted(list(p.objects.keys()))[::-1]
#for path in paths:
#    indices = get_generalized_indices(city, path)
#    print(path, indices, descend(city, path), [p.objects.get(index, None) for index in indices])
leaf_indices = sum([get_generalized_indices(city, path) for path in paths], [])
proof_indices = get_proof_indices(leaf_indices)
assert object_keys == proof_indices, (object_keys, proof_indices)
# p = SSZPartial(infer_type(city), branch2)
assert p.coords[0] == city.coords[0]
assert p.coords[1] == city.coords[1]
assert p.coords.root() == hash_tree_root(city.coords)
assert p.people[4].name[1] == city.people[4].name[1]
assert len(p.people[4].name) == len(city.people[4].name)
assert p.people[9].is_male == city.people[9].is_male
assert p.people[7].is_male == city.people[7].is_male
assert p.people[7].age == city.people[7].age
assert p.people[7].name[0] == city.people[7].name[0]
assert str(p.people[7].name) == str(city.people[7].name)
assert str(p.people[1]) == str(city.people[1])
assert p.people[1].name.root() == hash_tree_root(city.people[1].name)
assert p.root() == hash_tree_root(city)
print(hash_tree_root(city))
print("Tests passed")
