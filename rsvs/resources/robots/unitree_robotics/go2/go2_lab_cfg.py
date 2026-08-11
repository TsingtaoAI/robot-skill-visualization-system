from isaaclab.actuators import DCMotorCfg

GO2_ACTUATOR_CFG = {
    "hip_thigh": DCMotorCfg(
        joint_names_expr=[".*_hip_joint", ".*_thigh_joint"],
        effort_limit_sim=23.7,
        effort_limit=23.7,
        saturation_effort=23.7,
        velocity_limit_sim=30.1,
        velocity_limit=30.1,
        stiffness=0.0, # use self defined stiffness and damping
        damping=0.0,
        armature=0.01
    ),
    "calf": DCMotorCfg(
        joint_names_expr=[".*_calf_joint"],
        effort_limit_sim=45.43,
        effort_limit=45.43,
        saturation_effort=45.43,
        velocity_limit_sim=15.7,
        velocity_limit=15.7,
        stiffness=0.0, # use self defined stiffness and damping
        damping=0.0,
        armature=0.01
    )
}