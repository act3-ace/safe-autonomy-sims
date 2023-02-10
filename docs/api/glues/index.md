# glues

Glues connect an agent platform to a RL training framework
by providing an endpoint for observations and actions.

A glue may transform and send data from a platform part
to a training framework as an observation. The full set of
glues which provide observations over all agents in the
environment defines the environment's observation space.

A glue may transform and send data from a training
framework to a platform part as an action. The full set
of glues which provide actions over all agents in the
environment defines the environment's action space.

Multiple glues may be used in a decorator pattern to 
reuse common transformation logic. See [CoRL
documentation]({{corl_docs_url}}/reference/glues/__init__/) for more details!

- [Rejoin region glue](../../reference/glues/rejoin_region_glue.md)
- [Rta glue](../../reference/glues/rta_glue.md)
- [Vel limit glue](../../reference/glues/vel_limit_glue.md)
