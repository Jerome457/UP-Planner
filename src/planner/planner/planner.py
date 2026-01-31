import rclpy
from rclpy.node import Node
from std_msgs.msg import String

import json
import os
import math

from unified_planning import *
from unified_planning.shortcuts import *
import unified_planning.engines.results as results

from up_spiderplan.solver import EngineImpl
from up_spiderplan.util import plot_path


class PickPlacePlanner(Node):

    def __init__(self):
        super().__init__("pick_place_planner")

        self.subscription = self.create_subscription(
            String,
            "/parsed_object_tasks",
            self.task_callback,
            10
        )
        self.plan_pub = self.create_publisher(
            String,
            "/planned_actions",
            10
        )


        self.get_logger().info("Planner node started, waiting for tasks...")

    def task_callback(self, msg: String):
        self.get_logger().info("Received task assignment")
        tasks = json.loads(msg.data)

        problem = self.build_problem(tasks)
        self.solve_problem(problem)

    # ------------------------------------------------
    # BUILD PLANNING PROBLEM
    # ------------------------------------------------
    def build_problem(self, tasks):

        # -------------------------
        # TYPES
        # -------------------------
        t_robot = MovableType("robot")
        t_object = UserType("object")


        occ_map = OccupancyMap(
            "/home/jerome/planner/slam_map_clean.yaml",
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
        robot_at = Fluent("robot_at", BoolType(), robot=t_robot, configuration=t_robot_config)
        object_at = Fluent("object_at", BoolType(), obj=t_object, configuration=t_robot_config)
        holding = Fluent("holding", BoolType(), robot=t_robot, obj=t_object)
        handempty = Fluent("handempty", BoolType(), robot=t_robot)

        # -------------------------
        # WORKSTATIONS
        # -------------------------
        WS = {
            "WS01": ConfigurationObject("WS01", t_robot_config, (38.7368, 23.4332, 3 * math.pi / 2)),
            "WS02": ConfigurationObject("WS02", t_robot_config, (24.772, 41.8968, 0.0)),
            "WS04": ConfigurationObject("WS04", t_robot_config, (23.4742, 18.5907, 3 * math.pi / 2)),
        }

        # -------------------------
        # OBJECTS
        # -------------------------
        objects = {}
        for obj_id, _, _ in tasks:
            if obj_id not in objects:
                objects[obj_id] = Object(obj_id, t_object)

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
        # ACTIONS
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
        move.add_motion_constraint(Waypoints(robot, c_from, [c_to]))

        pick = InstantaneousAction("pick", robot=t_robot, obj=t_object, ws=t_robot_config)
        pr, po, pws = pick.parameters

        pick.add_precondition(robot_at(pr, pws))
        pick.add_precondition(object_at(po, pws))
        pick.add_precondition(handempty(pr))
        pick.add_effect(object_at(po, pws), False)
        pick.add_effect(holding(pr, po), True)
        pick.add_effect(handempty(pr), False)

        place = InstantaneousAction("place", robot=t_robot, obj=t_object, ws=t_robot_config)
        rr, ro, rws = place.parameters

        place.add_precondition(robot_at(rr, rws))
        place.add_precondition(holding(rr, ro))
        place.add_effect(holding(rr, ro), False)
        place.add_effect(object_at(ro, rws), True)
        place.add_effect(handempty(rr), True)

        # -------------------------
        # PROBLEM
        # -------------------------
        problem = Problem("TAMP_DYNAMIC")

        problem.add_fluent(robot_at, default_initial_value=False)
        problem.add_fluent(object_at, default_initial_value=False)
        problem.add_fluent(holding, default_initial_value=False)
        problem.add_fluent(handempty, default_initial_value=False)

        problem.add_action(move)
        problem.add_action(pick)
        problem.add_action(place)

        problem.add_objects(list(WS.values()))
        problem.add_objects(list(objects.values()))
        problem.add_object(r1)

        # -------------------------
        # INITIAL STATE
        # -------------------------
        problem.set_initial_value(robot_at(r1, WS["WS01"]), True)
        problem.set_initial_value(handempty(r1), True)

        for obj_id, src, _ in tasks:
            problem.set_initial_value(object_at(objects[obj_id], WS[src]), True)

        # -------------------------
        # GOALS
        # -------------------------
        for obj_id, _, dst in tasks:
            problem.add_goal(object_at(objects[obj_id], WS[dst]))

        return problem

    # ------------------------------------------------
    # SOLVE
    # ------------------------------------------------
    def solve_problem(self, problem):
        solver = EngineImpl(run_docker=True)
        result = solver.solve(problem)

        if result.status in results.POSITIVE_OUTCOMES:
            self.get_logger().info("PLAN FOUND")

            plan_text = self.plan_to_spiderplan_text(result.plan)

            msg = String()
            msg.data = plan_text
            self.plan_pub.publish(msg)

            self.get_logger().info("Plan published on /planned_actions")
            print(plan_text)

            plot_path(result)
        else:
            self.get_logger().error("NO PLAN FOUND")
            print(result.status)

    def plan_to_spiderplan_text(self, plan):
        lines = []

        for a in plan.actions:
            # This matches SpiderPlan console output
            lines.append(str(a))

        return "\n".join(lines)




def main():
    rclpy.init()
    node = PickPlacePlanner()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()



if __name__ == "__main__":
    main()
