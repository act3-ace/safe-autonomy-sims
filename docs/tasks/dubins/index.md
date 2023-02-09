# Dubins Rejoin

## Description

This task challenges an ML agent to learn to perform a basic maneuver
used in aircraft formation flight: the rejoin. Default rejoin environments
contain two aircraft, one lead and one wingman. The lead aircraft, by default, 
flies on a scripted path, while the wingman, controlled by an ML agent, attempts
to reach and maintain a relative position to the lead. The dynamics for each 
aircraft are given by the simple dubins aircraft model.
