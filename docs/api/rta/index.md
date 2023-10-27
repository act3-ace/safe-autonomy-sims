# rta

This package introduces Runtime Assurance modules for use
within the CoRL framework.

Runtime Assurance (RTA) is a safety technique which intercepts a
control or action before it is executed and checks it
against a well-defined safety constraint to see if the action
is safe. If the action is safe, the RTA module allows it
to execute. If the action is deemed **unsafe** the RTA module
provides an alternative action which is guaranteed to be safe
under the assumption that the system is already in a safe state.

- [cwh rta](../../reference/rta/cwh/cwh_rta.md)
- [inspection rta 1v1](../../reference/rta/cwh/inspection_rta_1v1.md)
