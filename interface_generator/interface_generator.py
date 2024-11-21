import binascii
import json

header_template = """
<script>
ethereum.enable()
c = web3.eth.contract(%s).at("%s")
</script>
"""

function_top_template = """
<b><span id="%s___start">%s</span></b>
<table>
"""

argument_template = """
<tr>
    <td><span>%s</span></td>
    <td><input id="%s"></input></td>
</tr>
"""

simple_prefill_template = """
<script>
document.getElementById("%s").value = "%s"
</script>
"""

prefill_by_call_script_template = """
<script>
setTimeout(function() {
    c.%s(%s function(err, res) {
        document.getElementById("%s").value = res.toString ? res.toString("10") : res;
    });
}, 1000)
</script>
"""

click_template = """
<script>
function ___%s() {
    c.%s(%s, function(err, res) { document.getElementById("%s___result").value = err+res; });
}
</script>
<tr>
    <td></td>
    <td><input type="submit" onclick="___%s()" /></td>
</tr>
"""

show_template = """
<tr>
    <td><span>%s(%s)</span></td>
    <td><span id="%s"></span></td>
</tr>
<script>
setInterval(function() {
    c.%s(%s function(err, res) {
        document.getElementById("%s").innerText = res.toString ? res.toString("10") : res;
    });
}, 1000)
</script>
"""

function_bottom_template = """
<tr>
    <td></td>
    <td><span id="%s___result"></td>
</tr>
</table>
"""

def generate_interface(address, abi, instructions):
    code = header_template % (json.dumps(abi), address)
    for page in instructions:
        function_name = page['fun']
        code += function_top_template % (function_name, function_name)
        abi_args = [x for x in abi if x.get("name", None) == function_name][0]
        for inp in abi_args["inputs"]:
            code += argument_template % (inp["name"], function_name + "___" + inp["name"])
        if abi_args.get("payable", None):
            code += argument_template % ("ETH amount to send", function_name + "___" + "ETH_AMOUNT_")
        for prefill in page.get('prefills', []):
            assert prefill['arg'] in [x['name'] for x in abi_args['inputs']]
            element_id = function_name + "___" + prefill['arg'] 
            if "value" in prefill:
                code += simple_prefill_template % (element_id, prefill["value"])
            else:
                code += prefill_by_call_script_template % (
                    prefill['fun'],
                    ','.join(map(str, prefill['inputs'] + [' '])),
                    element_id,
                )
        input_args = ''
        for inp in abi_args['inputs']:
            input_args += 'document.getElementById("' + function_name + "___" + inp["name"] + '").value, '
        if abi_args.get('payable', None):
            input_args += '{value: web3.toWei(document.getElementById("' + function_name + '___ETH_AMOUNT_").value)}, '
        code += click_template % (
            function_name,
            function_name,
            input_args,
            function_name,
            function_name
        )
        for i, show in enumerate(page.get('shows', [])):
            element_id = function_name + "__show__" + str(i)
            code += show_template % (
                show['fun'],
                ','.join(map(str, show['inputs'])),
                element_id,
                show['fun'],
                ','.join(map(str, show['inputs'] + [' '])),
                element_id,
            )
        code += function_bottom_template % (function_name)
    return code
