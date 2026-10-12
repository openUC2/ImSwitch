# Shared colour setup for the runner scripts. Sourced, never executed.
#
# pytest drops colour without a tty and neither ssh nor docker exec provides
# one here. Defines COLOR: the pytest flag on a terminal, empty otherwise. The
# scripts pass it unquoted, so the empty case expands to no argument at all.

COLOR=""
if [ -t 1 ]; then COLOR="--color=yes"; fi
