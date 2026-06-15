import genesis as gs
import numpy as np
import argparse

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("-v", "--vis", action="store_true", default=False)
    parser.add_argument("-c", "--cpu", action="store_true", default=False)
    args = parser.parse_args()

    gs.init(backend=gs.cpu if args.cpu else gs.gpu)

    scene = gs.Scene(
        sim_options=gs.options.SimOptions(dt=0.01),
        rigid_options=gs.options.RigidOptions(box_box_detection=True),
        show_viewer=args.vis,
    )
    plane = scene.add_entity(gs.morphs.Plane())
    cube = scene.add_entity(gs.morphs.Box(size=(0.04,0.04,0.04),pos=(0.65,0,0.02)),
                            surface=gs.surfaces.Plastic(color=(1,0,0)))
    cube_2 = scene.add_entity(gs.morphs.Box(size=(0.04,0.04,0.04),pos=(0.4,0.2,0.02)),
                              surface=gs.surfaces.Plastic(color=(0,1,0)))
    franka = scene.add_entity(gs.morphs.URDF(file="suction_robot/franka/franka_suction.urdf", fixed=True),
                              vis_mode="visual")

    scene.build()

    motors_dof = np.arange(7)
    kp = np.array([4500,4500,3500,3500,2000,2000,2000],dtype=np.float32)
    kv = np.array([450,450,350,350,200,200,200],dtype=np.float32)
    fmin = np.array([-87,-87,-87,-87,-12,-12,-12],dtype=np.float32)
    fmax = np.array([87,87,87,87,12,12,12],dtype=np.float32)
    franka.set_dofs_kp(kp=kp, dofs_idx_local=motors_dof)
    franka.set_dofs_kv(kv=kv, dofs_idx_local=motors_dof)
    franka.set_dofs_force_range(lower=fmin, upper=fmax, dofs_idx_local=motors_dof)

    end_effector = franka.get_link("link7")

    # move
    qpos = franka.inverse_kinematics(link=end_effector,
                                     pos=np.array([0.65,0,0.40]),
                                     quat=np.array([0,1,0,0]))
    path = franka.plan_path(qpos_goal=qpos, num_waypoints=100)
    for waypoint in path:
        franka.control_dofs_position(waypoint)
        scene.step()

    for _ in range(100):
        scene.step()

    def step_n(n):
        for _ in range(n):
            scene.step()

    step_n(100)
    rigid = scene.sim.rigid_solver
    link_cube = cube.get_link("box_baselink").idx
    link_franka = franka.get_link("link7").idx
    rigid.add_weld_constraint(link_cube, link_franka)

    # lift
    qpos = franka.inverse_kinematics(link=end_effector,pos=np.array([0.65,0,0.25]),quat=np.array([0,1,0,0]))
    franka.control_dofs_position(qpos)
    step_n(50)

    # reach
    qpos = franka.inverse_kinematics(link=end_effector,pos=np.array([0.65,0,0.37]),quat=np.array([0,1,0,0]))
    franka.control_dofs_position(qpos)
    step_n(50)

    # move
    qpos = franka.inverse_kinematics(link=end_effector,pos=np.array([0.4,0.2,0.45]),quat=np.array([0,1,0,0]))
    franka.control_dofs_position(qpos)
    step_n(50)

    # release
    rigid.delete_weld_constraint(link_cube, link_franka)
    step_n(200)

if __name__ == "__main__":
    main()
