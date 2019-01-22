### Interface generator

This is a quick-and-dirty draft of a medium-level interface-generating language targeting Ethereum ABI contracts. The goal is to allow a user to write an "interface file" that specifies some details about what the layout of an interface for a contract should look like, but in a highly restricted form that ensures that the interface cannot be misleading as long as the given contract code/ABI is "honest".

This is intended to reduce barriers to entry to using smart contracts by generating a "default interface" for any contract, and this code should ideally be ported to Javascript and put into an extension (plus a traditional centralized website), so users can enter something like eg. `http://coolwebsiteurl.fancytld/0xde0b295669a9fd93d5f28d9ec85e40f4cb697bae/my_interface_ptr` and have it pop up an automatically generated trustable interface for the contract.

There have already been experiments in automated interface generators that look at contract code alone, eg. for the contract above see https://etherscan.io/address/0xde0b295669a9fd93d5f28d9ec85e40f4cb697bae#readContract, but so far use of these has not taken off because contract code simply does not contain enough UI-critical information about what functions are more "important", what suggestions to make to users, etc. However, simply using Javascript to generate interfaces faces the problem that it is _too_ liberal, allowing a malicious interface designer to easily make a misleading interface. The goal here is to make a language that is in the middle, expressive enough to be somewhat usable in some circumstances, but not expressive enough to allow misleading users.

The author is allowed to create a list of "tabs", each tab corresponding to one function execution (the language could and should be extended to offer a "dashboard" tab that simply shows info without offering any functions to execute), with a textbox for each argument. The interface designer can suggest prefilled values for arguments, which can be either constant values or calls of a constant function of the same contract, and can also give information to the user in the form of calls of constant functions with preset arguments that are repeated once per second.

See examples/foundation_out.html in this repo for a simple example using the Ethereum Foundation multisig wallet (only seven people in the world can use this to do anything useful; it is intended for illustrative purposes only). See examples/foundation_interface.json for the interface file that produces this.

### TODOs

* Translate the interface generator from python to javascript so it can be run inside a web browser
* Allow interface files ("interface.json" here) to be written in YAML rather than JS 
* If the contract source code has variable-specific docstrings, display those beside the variable name and the textbox
* Allow input types other than textboxes, eg. dropdowns, sliders, and specialized text boxes for ETH or token values so users do not have to multiply by 10\*\*18 manually; this could be done via a "specialInputFormats" list in the interface similar to the "prefills" list.
* Allow outputs of shows to be represented in multiple ways.
