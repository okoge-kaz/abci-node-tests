## ABCI node tests

### Usage

```bash
git clone git@github.com:okoge-kaz/abci-node-tests.git
cd abci-node-tests

# v100 node
qsub -g <group> -ar <reserved_id> scripts/v100_node_check.sh

# a100 node
qsub -g <group> -ar <reserved_id> scripts/a100_node_check.sh
```

