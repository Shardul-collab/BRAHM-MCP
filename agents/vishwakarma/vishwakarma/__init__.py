# vishwakarma/__init__.py
#
# Vishwakarma — Quantum ESPRESSO agent for BRAHM MCP
#
# Each module exposes a clean functional API consumed by mcp_server.py.
# No HTTP, no cloud. All execution is local via subprocess.
#
# Module map:
#   input_generator  — build QE input files (.in) for any calc type
#   runner           — execute QE binaries, manage job directories
#   output_parser    — extract structured data from QE output files
#   pseudo_manager   — discover and validate UPF pseudopotential files
#   workflow         — orchestrate multi-step calculation sequences
#   calculators/     — per-code wrappers (pw, ph, pp, dos, bands, neb)
