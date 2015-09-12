import casper
import sys

casper.logging_level = int(sys.argv[1]) if len(sys.argv) > 1 else 0
casper.run(50000)
