#!/bin/bash
FADMEN_HOME=$(dirname $PWD)
(sed -e "s@<<FADMEN_HOME>>@$FADMEN_HOME@" < SingularityTemplate) > FADMEN.def
sudo singularity build FADMEN.sif FADMEN.def
