"""
This module defines tests for the CWHSpacecraft entity.

Author: John McCarroll
"""

import pytest
import numpy as np

from saferl_sim.cwh.cwh import CWHSpacecraft
from tests.test_simulators.test_backend_simulators.conftest import evaluate


# override entity fixture
@pytest.fixture
def entity(initial_entity_state):
    entity = CWHSpacecraft(name="tests")

    if initial_entity_state is not None:
        entity.state = initial_entity_state

    return entity


# Define tests assay
test_configs = [
    # No Control
    # - At origin stationary
    (
        np.array([0, 0, 0, 0, 0, 0]),                           # initial_entity_state
        {},                                                     # action
        10,                                                     # num_steps
        {'state': np.array([0, 0, 0, 0, 0, 0])},                # attr_targets
        0.1                                                     # error_bound
    ),

    # - At origin moving positive x
    (
        np.array([0, 0, 0, 2, 0, 0]),                           # initial_entity_state
        {},                                                     # action
        10,                                                     # num_steps
        {'state': np.array(                                     # attr_targets
            [19.999648425520753,
             -0.2053981946620434,
             0.0,
             1.999894528027041,
             -0.04107927786601962,
             0.0]
        )},
        0.1                                                     # error_bound
    ),

    # - At origin moving negative x
    (
        np.array([0, 0, 0, -2, 0, 0]),                           # initial_entity_state
        {},                                                     # action
        10,                                                     # num_steps
        {'state': np.array(                                     # attr_targets
            [19.999648425520753,
             0.2053981946620434,
             0.0,
             -1.999894528027041,
             0.04107927786601962,
             0.0]
        )},
        0.1                                                     # error_bound
    ),

    # - At origin moving positive y
    (
        np.array([0, 0, 0, 0, 2, 0]),                          # initial_entity_state
        {},                                                     # action
        10,                                                     # num_steps
        {'state': np.array(                                     # attr_targets
            [0.2053981946620434,
             19.99859370208301,
             0.0,
             0.04107927786601962,
             1.9995781121081642,
             0.0]
        )},
        0.1                                                     # error_bound
    ),

    # - At origin moving negative y
    (
        np.array([0, 0, 0, 0, -2, 0]),                          # initial_entity_state
        {},                                                     # action
        10,                                                     # num_steps
        {'state': np.array(                                     # attr_targets
            [-0.2053981946620434,
             -19.99859370208301,
             0.0,
             -0.04107927786601962,
             -1.9995781121081642,
             0.0]
        )},
        0.1                                                     # error_bound
    ),

    # - At origin moving positive z
    (
        np.array([0, 0, 0, 0, 0, 2]),                           # initial_entity_state
        {},                                                     # action
        10,                                                     # num_steps
        {'state': np.array(                                     # attr_targets
            [0.0,
             0.0,
             19.999648425520753,
             0.0,
             0.0,
             1.999894528027041]
        )},
        0.1                                                     # error_bound
    ),

    # - At origin moving negative z
    (
        np.array([0, 0, 0, 0, 0, -2]),                          # initial_entity_state
        {},                                                     # action
        10,                                                     # num_steps
        {'state': np.array(                                     # attr_targets
            [0.0,
             0.0,
             -19.999648425520753,
             0.0,
             0.0,
             -1.999894528027041]
        )},
        0.1                                                     # error_bound
    ),

    # - At origin moving all directions
    (
        np.array([0, 0, 0, 3, -4, 8]),                          # initial_entity_state
        {},                                                     # action
        10,                                                     # num_steps
        {'state': np.array(                                     # attr_targets
            [29.588676248957043,
             -40.305284696159084,
             79.99859370208301,
             2.9176832363085223,
             -4.060775141015358,
             7.999578112108164]
        )},
        0.1                                                     # error_bound
    ),

    # - At positive point 1 stationary
    (
        np.array([1795, 293, 590, 0, 0, 0]),                    # initial_entity_state
        {},                                                     # action
        10,                                                     # num_steps
        {'state': np.array(                                     # attr_targets
            [1795.283983287192,
             292.9980556542578,
             589.9688857679771,
             0.05679615822846139,
             -0.0005833016718923634,
             -0.0062227917093393165]
        )},
        0.1                                                     # error_bound
    ),

    # - At positive point 2 stationary
    (
        np.array([-424, -342, -748, 0, 0, 0]),                  # initial_entity_state
        {},                                                     # action
        10,                                                     # num_steps
        {'state': np.array(                                     # attr_targets
            [-424.0670801748019,
             -341.9995407227885,
             -747.9605534821134,
             -0.013415917041151883,
             0.00013778267904309866,
             0.007889234234891202]
        )},
        0.1                                                     # error_bound
    ),

    # - At positive point 3 stationary
    (
        np.array([-241, 1932, 738, 0, 0, 0]),                   # initial_entity_state
        {},                                                     # action
        10,                                                     # num_steps
        {'state': np.array(                                     # attr_targets
            [-241.03812811822468,
             1932.0002610514339,
             737.9610808419782,
             -0.007625556620088688,
             7.831515483345938e-05,
             -0.007783763188970196]
        )},
        0.1                                                     # error_bound
    ),

    # - At positive point 1 moving
    (
        np.array([1795, 293, 590, 2.81649526,  -4.41614603, -19.03240191]),  # initial_entity_state
        {},                                                     # action
        10,                                                     # num_steps
        {'state': np.array(                                     # attr_targets
            [1822.994906572302,
             248.55044904190027,
             399.6482123213723,
             2.7824368426408657,
             -4.4736475680995085,
             -19.037621009219546]
        )},
        0.1                                                     # error_bound
    ),

    # - At positive point 2 moving
    (
        np.array([-424, -342, -748, 3.3033103, -17.06746379, 10.78131432]),  # initial_entity_state
        {},                                                     # action
        10,                                                     # num_steps
        {'state': np.array(                                     # attr_targets
            [-392.78737097956406,
             -513.0014246394147,
             -640.1493054995972,
             2.9391606358827103,
             -17.131574530007974,
             10.788634990988681]
        )},
        0.1                                                     # error_bound
    ),

    # - At positive point 3 moving
    (
        np.array([-241, 1932, 738, -11.31675638, -14.37759259, 8.03893148]),  # initial_entity_state
        {},                                                     # action
        10,                                                     # num_steps
        {'state': np.array(                                     # attr_targets
            [-355.68026835764533,
             1789.3966654056303,
             818.3489825004037,
             -11.61909569683316,
             -14.142039318793397,
             8.030723775829191]
        )},
        0.1                                                     # error_bound
    ),

    # With Control
    # - At origin stationary max control
    (
        np.array([0, 0, 0, 0, 0, 0]),                           # initial_entity_state
        {'thrust_x': 1, 'thrust_y': 1, 'thrust_z': 1},          # action
        10,                                                     # num_steps
        {'state': np.array(                                     # attr_targets
            [4.195157671606051,
             4.137992549760965,
             4.1666300442734014,
             0.8418769425076355,
             0.8247164794758544,
             0.8333186843966979]
        )},
        0.1                                                     # error_bound
    ),

    # - At origin stationary min control
    (
        np.array([0, 0, 0, 0, 0, 0]),                           # initial_entity_state
        {'thrust_x': -1, 'thrust_y': 1, 'thrust_z': -1},        # action
        10,                                                     # num_steps
        {'state': np.array(                                     # attr_targets
            [-4.195157671606051,
             -4.137992549760965,
             -4.1666300442734014,
             -0.8418769425076355,
             -0.8247164794758544,
             -0.8333186843966979]
        )},
        0.1                                                     # error_bound
    ),

    # - At origin stationary max control exceeded
    (
        np.array([0, 0, 0, 0, 0, 0]),                           # initial_entity_state
        {'thrust_x': 1.1, 'thrust_y': 1.1, 'thrust_z': 1.1},    # action
        10,                                                     # num_steps
        {'state': np.array(                                     # attr_targets
            [4.195157671606051,
             4.137992549760965,
             4.1666300442734014,
             0.8418769425076355,
             0.8247164794758544,
             0.8333186843966979]
        )},
        0.1                                                     # error_bound
    ),

    # - At origin stationary min control exceeded
    (
        np.array([0, 0, 0, 0, 0, 0]),                               # initial_entity_state
        {'thrust_x': -1.1, 'thrust_y': -1.1, 'thrust_z': -1.1},     # action
        10,                                                         # num_steps
        {'state': np.array(                                         # attr_targets
            [-4.195157671606051,
             -4.137992549760965,
             -4.1666300442734014,
             -0.8418769425076355,
             -0.8247164794758544,
             -0.8333186843966979]
        )},
        0.1                                                         # error_bound
    ),

    # - random inputs 1
    (
        np.array(                                                   # initial_entity_state
            [378.43311614769686,                                    # - initial position
             -899.1465249919404,
             48.827473316718624,
             7.373886360503278,                                     # - initial velocity
             -5.387059196674235,
             -1.605257585437565]
        ),
        {                                                           # action
            'thrust_x': -0.08432175853896218,
            'thrust_y': -1.1,
            'thrust_z': -1.1
        },
        15,                                                         # num_steps
        {'state': np.array(                                         # attr_targets
            [487.1005400235034,                                     # - final position
             -985.1335654807021,
             18.711398036269838,
             7.1124226210522785,                                    # - final velocity
             -6.076536913524862,
             -2.4101397075543263]
        )},
        0.1                                                         # error_bound
    ),

    # - random inputs 2
    (
        np.array(                                                   # initial_entity_state
            [401.0710014944277,                                     # - initial position
             -524.8182691047782,
             125.17724674622104,
             -9.575557314121877,                                    # - initial velocity
             6.880063127211869,
             -5.5428664275674695]
        ),
        {                                                           # action
            'thrust_x': -0.27372446784455806,
            'thrust_y': -0.4075755672077819,
            'thrust_z': 0.7043004899796548
        },
        59,                                                         # num_steps
        {'state': np.array(                                         # attr_targets
            [-178.82135098778141,                                   # - final position
             -143.1947617073997,
             -99.75985244440577,
             -10.116272691726516,                                   # - final velocity
             6.067248813772067,
             -2.0797869085758034]
        )},
        0.1                                                         # error_bound
    ),

    # - random inputs 3
    (
        np.array(                                                   # initial_entity_state
            [-523.1457315141807,
             -248.7512703155934,
             904.8316967095891,
             5.2410919696930165,
             -5.7675255661019715,
             -9.383970813566798]
        ),
        {                                                           # action
            'thrust_x': -0.601994261048153,
            'thrust_y': -0.6380545923556369,
            'thrust_z': 0.5967382067922504
        },
        59,                                                         # num_steps
        {'state': np.array(                                         # attr_targets
            [-326.1945733785029,
             -714.4418539210326,
             423.4978690814901,
             1.1464758782969273,
             -9.415507422720474,
             -6.392304440881457]
        )},
        0.1                                                         # error_bound
    ),
]


@pytest.mark.parametrize("initial_entity_state,action,num_steps,attr_targets,error_bound", test_configs, indirect=True)
def test_CWHSpacecraft(entity, action, num_steps, attr_targets, error_bound):
    evaluate(entity, attr_targets, error_bound=error_bound)
