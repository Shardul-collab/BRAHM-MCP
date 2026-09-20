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
#
# The tool surface consumed by BRAHM lives in brahm/agents/vishwakarma.py
# (registered via brahm_registry) and in vishwakarma_api.py. The in-package
# mcp_server.py is a v1 monolith left over from the pre-registry design —
# it is imported by nothing and brahm_audit.py flags it as a "v1 ghost".
#
# A calculators/ package (per-code wrappers) was planned and listed here but
# never written; the empty directory was removed 2026-09-09. Per-code logic
# lives in input_generator/output_parser instead, keyed by code name.
