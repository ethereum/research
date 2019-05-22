from hashable_list import HashableList
import binascii

def hex(x): return binascii.hexlify(x).decode('utf-8')

input_values = [(2**31 + x).to_bytes(4, 'little') for x in range(30)]

h = HashableList(input_values)

print("Generated tree, root: %s" % hex(h.root))

assert [h[i] for i in range(len(input_values))] == input_values

print("Getters work")
print("Values (first 10):", input_values[:10])

print("Set h[2] = 8080")
eightyeighty = (8080).to_bytes(4, 'little')
h[2] = eightyeighty

assert int.from_bytes(h[2], 'little') == 8080
print("Setter works")

h2 = HashableList(input_values[:2] + [eightyeighty] + input_values[3:])

assert h.root == h2.root
print("Root matches")

new_input_values = input_values[:2] + [eightyeighty] + input_values[3:]

for i in range(30, 70):
    print(i)
    newval = (2**31 + i).to_bytes(4, 'little')
    h.append(newval)
    new_input_values.append(newval)
    h_new = HashableList(new_input_values)
    assert h.root == h_new.root

print('Append works! Going backwards')

for i in range(69, 29, -1):
    print(i)
    h.pop()
    new_input_values.pop()
    h_new = HashableList(new_input_values)
    assert h.root == h_new.root

print('Pop works! One more roller coaster ride up!')

for i in range(30, 70):
    print(i)
    newval = (2**16 + i).to_bytes(4, 'little')
    h.append(newval)
    new_input_values.append(newval)
    h_new = HashableList(new_input_values)
    assert h.root == h_new.root

print('Append still works!')
