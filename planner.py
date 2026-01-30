from unified_planning import *
from unified_planning.shortcuts import *
import unified_planning.engines.results as results
import os
import math

# -------------------------
# TYPES
# -------------------------
t_robot = MovableType("robot")
t_object = UserType("object")

occ_map = OccupancyMap(
    os.path.join("./slam_map_clean.yaml"),
    (0, 0)
)

t_robot_config = ConfigurationType(
    "robot_config",
    occ_map,
    3
)

# -------------------------
# FLUENTS
# -------------------------
robot_at = Fluent(
    "robot_at",
    BoolType(),
    robot=t_robot,
    configuration=t_robot_config
)

object_at = Fluent(
    "object_at",
    BoolType(),
    obj=t_object,
    configuration=t_robot_config
)

holding = Fluent(
    "holding",
    BoolType(),
    robot=t_robot,
    obj=t_object
)

handempty = Fluent(
    "handempty",
    BoolType(),
    robot=t_robot
)

# -------------------------
# WORKSTATIONS
# -------------------------
WS01 = ConfigurationObject(
    "WS01",
    t_robot_config,
    (38.7368, 23.4332, 3 * math.pi / 2)
)

WS02 = ConfigurationObject(
    "WS02",
    t_robot_config,
    (24.772, 41.8968, 0.0)
)

WS04 = ConfigurationObject(
    "WS04",
    t_robot_config,
    (23.4742, 18.5907, 3 * math.pi / 2)
)

# -------------------------
# OBJECTS
# -------------------------
obj11 = Object("obj11", t_object)
obj12 = Object("obj12", t_object)
obj13 = Object("obj13", t_object)
obj16 = Object("obj16", t_object)

# -------------------------
# ROBOT
# -------------------------
r1 = MovableObject(
    "robot-1",
    t_robot,
    footprint=[
        (-0.25,  0.25),
        ( 0.25,  0.25),
        ( 0.25, -0.25),
        (-0.25, -0.25),
    ],
    motion_model=MotionModels.REEDSSHEPP,
    parameters={"turning_radius": 0.5},
)

# -------------------------
# MOVE ACTION
# -------------------------
move = InstantaneousMotionAction(
    "move",
    robot=t_robot,
    c_from=t_robot_config,
    c_to=t_robot_config
)

robot = move.parameter("robot")
c_from = move.parameter("c_from")
c_to = move.parameter("c_to")

move.add_precondition(robot_at(robot, c_from))
move.add_effect(robot_at(robot, c_from), False)
move.add_effect(robot_at(robot, c_to), True)

move.add_motion_constraint(
    Waypoints(robot, c_from, [c_to])
)

# -------------------------
# PICK ACTION
# -------------------------
pick = InstantaneousAction(
    "pick",
    robot=t_robot,
    obj=t_object,
    ws=t_robot_config
)

pr = pick.parameter("robot")
po = pick.parameter("obj")
pws = pick.parameter("ws")

pick.add_precondition(robot_at(pr, pws))
pick.add_precondition(object_at(po, pws))
pick.add_precondition(handempty(pr))

pick.add_effect(object_at(po, pws), False)
pick.add_effect(holding(pr, po), True)
pick.add_effect(handempty(pr), False)

# -------------------------
# PLACE ACTION
# -------------------------
place = InstantaneousAction(
    "place",
    robot=t_robot,
    obj=t_object,
    ws=t_robot_config
)

rr = place.parameter("robot")
ro = place.parameter("obj")
rws = place.parameter("ws")

place.add_precondition(robot_at(rr, rws))
place.add_precondition(holding(rr, ro))

place.add_effect(holding(rr, ro), False)
place.add_effect(object_at(ro, rws), True)
place.add_effect(handempty(rr), True)

# -------------------------
# PROBLEM
# -------------------------
problem = Problem("TAMP_PICK_PLACE")

problem.add_fluent(robot_at, default_initial_value=False)
problem.add_fluent(object_at, default_initial_value=False)
problem.add_fluent(holding, default_initial_value=False)
problem.add_fluent(handempty, default_initial_value=False)

problem.add_action(move)
problem.add_action(pick)
problem.add_action(place)

problem.add_objects([WS01, WS02, WS04])
problem.add_objects([obj11, obj12, obj13, obj16])
problem.add_object(r1)

# -------------------------
# INITIAL STATE
# -------------------------
problem.set_initial_value(robot_at(r1, WS01), True)
problem.set_initial_value(handempty(r1), True)

problem.set_initial_value(object_at(obj11, WS01), True)
problem.set_initial_value(object_at(obj13, WS01), True)
problem.set_initial_value(object_at(obj12, WS02), True)
problem.set_initial_value(object_at(obj16, WS04), True)

# -------------------------
# GOALS
# -------------------------
problem.add_goal(object_at(obj11, WS04))
problem.add_goal(object_at(obj13, WS04))
problem.add_goal(object_at(obj12, WS04))
problem.add_goal(object_at(obj16, WS02))

# -------------------------
# SOLVE (SPIDERPLAN)
# -------------------------
from up_spiderplan.solver import EngineImpl
from up_spiderplan.util import plot_path

solver = EngineImpl(run_docker=True)
result = solver.solve(problem)

# -------------------------
# OUTPUT
# -------------------------
if result.status in results.POSITIVE_OUTCOMES:
    print("\n--- PLAN FOUND ---\n")
    for a in result.plan.actions:
        print(a)
        # if hasattr(a, "motion_paths"):
        #     print("Motion path:", a.motion_paths)
    plot_path(result)
else:
    print(result.status)
    print("\n--- NO PLAN FOUND ---\n")
