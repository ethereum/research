from mk_sendmany import mk_multisend_code, get_multisend_gas
from ethereum.tools import tester2
from ethereum import utils

# from ethereum.slogging import get_logger, configure_logging 
# logger = get_logger() 
# configure_logging(':trace')

c = tester2.Chain()
args = {utils.int_to_addr(10000 + i ** 3): i ** 3 for i in range(1, 101)}
c.tx(to=b'', value=sum(args.values()), data=mk_multisend_code(args), startgas=get_multisend_gas(args), gasprice=20 * 10**9)
for addr, value in args.items():
    assert c.head_state.get_balance(addr) == value
print("Test successful")
