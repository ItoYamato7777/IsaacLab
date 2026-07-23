#!/usr/bin/env python3

# Copyright (c) 2022-2024, NVIDIA CORPORATION. All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto. Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

"""
This script demonstrates how to create a cable/rope mechanism using individual prims.
The rope is made of rigid body capsules connected with spherical joints.

実行例:
    python3 rope_demo.py
"""

from isaacsim import SimulationApp

# Launch Isaac Sim
simulation_app = SimulationApp({"headless": False})

import time
from pxr import Usd, UsdGeom, UsdPhysics, Gf, Sdf, PhysxSchema, UsdShade, UsdLux
from omni.physx.scripts import physicsUtils
from omni.timeline import get_timeline_interface
from isaacsim.core.api import World
from isaacsim.core.utils.prims import get_prim_at_path

# Set Light stage to Grey Studio
stage = simulation_app.context.get_stage()
light_stage = stage.GetPrimAtPath("/Environment")
if not light_stage:
    light_stage = stage.DefinePrim("/Environment", "Xform")

# Create Grey Studio lighting environment
dome_light = UsdLux.DomeLight.Define(stage, "/Environment/GreyStudio")
dome_light.CreateIntensityAttr().Set(1000.0)

class SphericalJoint:
    def __init__(self, stage, joint_path, body0_path, body1_path, local_pos0, local_pos1,
                 damping=0.0, stiffness=0.0, enable_drives=True):
        self.stage = stage
        self.joint_path = joint_path
        self.body0_path = body0_path
        self.body1_path = body1_path
        self.local_pos0 = local_pos0
        self.local_pos1 = local_pos1
        self.damping = damping
        self.stiffness = stiffness
        self.enable_drives = enable_drives
        self.joint = None
        self._create_joint()

    def _create_joint(self):
        # Create a D6 joint
        self.joint = UsdPhysics.Joint.Define(self.stage, self.joint_path)
        joint_prim = self.joint.GetPrim()

        # Lock all translational degrees of freedom
        for axis in ["transX", "transY", "transZ", "rotX"]:
            limitAPI = UsdPhysics.LimitAPI.Apply(joint_prim, axis)
            limitAPI.CreateLowAttr(1.0)
            limitAPI.CreateHighAttr(-1.0)

        # Set the bodies to connect
        self.joint.GetBody0Rel().SetTargets([self.body0_path])
        self.joint.GetBody1Rel().SetTargets([self.body1_path])

        # Set local positions
        self.joint.CreateLocalPos0Attr().Set(self.local_pos0)
        self.joint.CreateLocalPos1Attr().Set(self.local_pos1)

        # Set local rotations (identity quaternions)
        self.joint.CreateLocalRot0Attr().Set(Gf.Quatf(1.0))
        self.joint.CreateLocalRot1Attr().Set(Gf.Quatf(1.0))

        # Configure joint drives for Y and Z rotation if enabled
        if self.enable_drives:
            for axis in ["rotY", "rotZ"]:
                limitAPI = UsdPhysics.LimitAPI.Apply(joint_prim, axis)
                limitAPI.CreateLowAttr(-1)
                limitAPI.CreateHighAttr(1)

                driveAPI = UsdPhysics.DriveAPI.Apply(joint_prim, axis)
                driveAPI.CreateTypeAttr("force")
                driveAPI.CreateDampingAttr(self.damping)
                driveAPI.CreateStiffnessAttr(self.stiffness)

    def get_joint(self):
        return self.joint

class RopeCreator:
    def __init__(self, stage, default_prim_path, physics_material_path, pivot_point=None,
                 link_half_length=0.1, rope_length=2.0, num_ropes=1, rope_spacing=0.1,
                 rope_color=None, rope_damping=2.0, rope_stiffness=0.0, contact_offset=0.01,
                 enable_joint_drives=True, rope_segment_mass=0.00005):
        self.stage = stage
        self.default_prim_path = default_prim_path
        self.physics_material_path = physics_material_path
        self.pivot_point = pivot_point if pivot_point is not None else Gf.Vec3f(0.0, 0.0, 1.0)

        # Configure rope parameters
        self.link_half_length = link_half_length
        self.link_radius = 0.5 * self.link_half_length
        self.rope_length = rope_length
        self.num_ropes = num_ropes
        self.rope_spacing = rope_spacing
        self.rope_color = rope_color if rope_color is not None else Gf.Vec3f(0.2, 0.6, 0.8)  # Blue-ish color
        self.rope_damping = rope_damping
        self.rope_stiffness = rope_stiffness
        self.contact_offset = contact_offset
        self.enable_joint_drives = enable_joint_drives
        self.rope_segment_mass = rope_segment_mass

        # Store segment prims and joints for the rope
        self.segments = []  # List of segment prims
        self.joints = []    # List of joint prims

    def create_capsule(self, path, position):
        capsule_geom = UsdGeom.Capsule.Define(self.stage, path)
        capsule_geom.CreateHeightAttr(self.link_half_length)
        capsule_geom.CreateRadiusAttr(self.link_radius)
        capsule_geom.CreateAxisAttr("X")
        capsule_geom.CreateDisplayColorAttr().Set([self.rope_color])

        # Set position
        capsule_geom.AddTranslateOp().Set(position)

        UsdPhysics.CollisionAPI.Apply(capsule_geom.GetPrim())
        UsdPhysics.RigidBodyAPI.Apply(capsule_geom.GetPrim())
        mass_api = UsdPhysics.MassAPI.Apply(capsule_geom.GetPrim())
        mass_api.CreateMassAttr().Set(self.rope_segment_mass)
        physx_collision_api = PhysxSchema.PhysxCollisionAPI.Apply(capsule_geom.GetPrim())
        physx_collision_api.CreateRestOffsetAttr().Set(0.0)
        physx_collision_api.CreateContactOffsetAttr().Set(self.contact_offset)
        physicsUtils.add_physics_material_to_prim(self.stage, capsule_geom.GetPrim(), self.physics_material_path)

        return capsule_geom

    def create_joint(self, joint_path, body0_path, body1_path, local_pos0, local_pos1):
        # Create a spherical joint using the new abstraction
        spherical_joint = SphericalJoint(
            self.stage,
            joint_path,
            body0_path,
            body1_path,
            local_pos0,
            local_pos1,
            damping=self.rope_damping,
            stiffness=self.rope_stiffness,
            enable_drives=self.enable_joint_drives
        )
        return spherical_joint.get_joint()

    def create_ropes(self):
        # Calculate spacing and dimensions
        link_length = 2.0 * self.link_half_length
        num_links = int(self.rope_length / link_length)
        y_start = -(self.num_ropes // 2) * self.rope_spacing

        for rope_ind in range(self.num_ropes):
            scope_path = self.default_prim_path.AppendChild(f"Rope{rope_ind}")
            UsdGeom.Scope.Define(self.stage, scope_path)

            y = y_start + rope_ind * self.rope_spacing
            z = self.pivot_point[2]  # Use pivot point's z coordinate

            # Create all capsule links
            for link_ind in range(num_links):
                x = self.pivot_point[0] + link_ind * link_length  # Start from pivot point's x coordinate
                position = Gf.Vec3f(x, self.pivot_point[1] + y, z)  # Add y offset for multiple ropes

                capsule_path = scope_path.AppendChild(f"capsule_{link_ind}")
                capsule_geom = self.create_capsule(capsule_path, position)
                self.segments.append(capsule_geom.GetPrim())

            # Create joints between consecutive capsules
            joint_x = self.link_half_length
            for link_ind in range(num_links - 1):
                joint_path = scope_path.AppendChild(f"joint_{link_ind}")

                body0_path = self.segments[link_ind].GetPath()
                body1_path = self.segments[link_ind + 1].GetPath()

                local_pos0 = Gf.Vec3f(joint_x, 0, 0)
                local_pos1 = Gf.Vec3f(-joint_x, 0, 0)

                joint = self.create_joint(joint_path, body0_path, body1_path, local_pos0, local_pos1)
                self.joints.append(joint)

    def get_segments(self):
        """Get all segment prims for the rope"""
        return self.segments

    def get_joints(self):
        """Get all joint prims for the rope"""
        return self.joints

    def get_first_segment(self):
        """Get the first segment prim for the rope"""
        return self.segments[0] if self.segments else None

    def get_last_segment(self):
        """Get the last segment prim for the rope"""
        return self.segments[-1] if self.segments else None

def create_cylinder(stage, parent_path, name, radius=0.5, height=1.0, position=None, rotation_y=90.0):
    """
    Create a cylinder with physics properties and a revolute joint.

    Args:
        stage: The USD stage
        parent_path: Path to the parent prim
        name: Name for the cylinder prim
        radius: Radius of the cylinder in meters
        height: Height of the cylinder in meters
        position: Position of the cylinder (Gf.Vec3f), defaults to (0, 0, 2)
        rotation_y: Rotation around Y axis in degrees, defaults to 90

    Returns:
        tuple: (cylinder_path, joint_path) - Paths to the created cylinder and joint
    """
    if position is None:
        position = Gf.Vec3f(0.0, 0.0, 1.0)

    # Create cylinder
    cylinder_path = parent_path.AppendChild(name)
    cylinder_geom = UsdGeom.Cylinder.Define(stage, cylinder_path)
    cylinder_geom.CreateRadiusAttr().Set(radius)
    cylinder_geom.CreateHeightAttr().Set(height)
    cylinder_geom.CreateAxisAttr().Set(UsdGeom.Tokens.z)

    # Set position and rotation
    cylinder_geom.AddTranslateOp().Set(position)
    cylinder_geom.AddRotateYOp().Set(rotation_y)

    # Add physics properties
    UsdPhysics.CollisionAPI.Apply(cylinder_geom.GetPrim())
    UsdPhysics.RigidBodyAPI.Apply(cylinder_geom.GetPrim())

    # Create revolute joint
    joint_path = parent_path.AppendChild(f"{name}_joint")
    joint = UsdPhysics.RevoluteJoint.Define(stage, joint_path)

    # Set joint bodies
    joint.CreateBody0Rel().SetTargets([parent_path])  # World
    joint.CreateBody1Rel().SetTargets([cylinder_path])  # Cylinder

    # Set joint axis
    joint.CreateAxisAttr().Set("Y")

    # Set local positions
    joint.CreateLocalPos0Attr().Set(position)
    joint.CreateLocalPos1Attr().Set(Gf.Vec3f(0.0, 0.0, 0.0))

    # Set local rotations
    joint.CreateLocalRot0Attr().Set(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
    joint.CreateLocalRot1Attr().Set(Gf.Quatf(0.707, 0.707, 0.0, 0.0))

    # Add joint drive
    drive = UsdPhysics.DriveAPI.Apply(joint.GetPrim(), "angular")
    drive.CreateTypeAttr().Set("force")
    drive.CreateTargetVelocityAttr().Set(1.0)  # Rotate at 1 rad/s
    drive.CreateDampingAttr().Set(1.0)
    drive.CreateStiffnessAttr().Set(10.0)

    return cylinder_path, joint_path, drive


# Create a world
my_world = World(stage_units_in_meters=1.0)

# Add default ground plane
my_world.scene.add_default_ground_plane()

# Get the stage
stage = simulation_app.context.get_stage()

# Enable Physics
scene = UsdPhysics.Scene.Define(stage, "/physicsScene")
scene.CreateGravityDirectionAttr().Set(Gf.Vec3f(0.0, 0.0, -1.0))
scene.CreateGravityMagnitudeAttr().Set(9.81)

# Create default prim and physics material
default_prim_path = Sdf.Path("/World")
UsdGeom.Xform.Define(stage, default_prim_path)
stage.SetDefaultPrim(stage.GetPrimAtPath(default_prim_path))

# Create the cylinder using the helper function
cylinder_radius = 0.3
cylinder_height = 2.0
cylinder_path, joint_path, drive = create_cylinder(stage, default_prim_path, "cylinder", radius=cylinder_radius, height=cylinder_height)


# Create the rope creator and generate ropes
physics_material_path = default_prim_path.AppendChild("PhysicsMaterial")
UsdShade.Material.Define(stage, physics_material_path)
material = UsdPhysics.MaterialAPI.Apply(stage.GetPrimAtPath(physics_material_path))
material.CreateStaticFrictionAttr().Set(0.5)
material.CreateDynamicFrictionAttr().Set(0.5)
material.CreateRestitutionAttr().Set(0)

rope_creator = RopeCreator(stage,
                           default_prim_path,
                           physics_material_path,
                           pivot_point=Gf.Vec3f(0.5, 0.0, 1.0),
                           rope_length=4.0,
                           num_ropes=1,
                           rope_spacing=0.1,
                           rope_damping=10e6,
                           rope_stiffness=0,
                           rope_segment_mass=10e-6,
                           enable_joint_drives=False)  # Set to False to disable joint drives
rope_creator.create_ropes()

# Create a spherical joint between the cylinder perimeter and the first capsule of the rope
# Get the first capsule path
first_capsule_path = "/World/Rope0/capsule_0"

# Create the spherical joint
attachment_joint_path = default_prim_path.AppendChild("cylinder_rope_attachment")
attachment_joint = UsdPhysics.SphericalJoint.Define(stage, attachment_joint_path)

# Connect the cylinder and first capsule
attachment_joint.GetBody0Rel().SetTargets([cylinder_path])
attachment_joint.GetBody1Rel().SetTargets([first_capsule_path])

# Set local positions - attach at cylinder perimeter (radius 0.5) and capsule center
attachment_joint.CreateLocalPos0Attr().Set(Gf.Vec3f(cylinder_radius, 0.0, 0.0))  # At cylinder perimeter
attachment_joint.CreateLocalPos1Attr().Set(Gf.Vec3f(-0.1, 0.0, 0.0))  # At capsule edge

# Set local rotations (identity quaternions)
attachment_joint.CreateLocalRot0Attr().Set(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
attachment_joint.CreateLocalRot1Attr().Set(Gf.Quatf(1.0, 0.0, 0.0, 0.0))


# Initial setup for the simulation
timeline = get_timeline_interface()

# Start the simulation
timeline.play()

# Wait for the simulation to start
while not timeline.is_playing():
    simulation_app.update()
    time.sleep(0.01)

# Keep the simulation running
print("Simulation started. Press Ctrl+C to exit")
position = 0.0

time_iter_to_cut = 5*100
loop_counter = 0
try:
    while timeline.is_playing():
        loop_counter += 1
        # Continuously increment the joint drive position
        position += 1.  # Increment by 0.01 radians per frame
        drive.CreateTargetPositionAttr().Set(position)

        # if loop_counter == time_iter_to_cut:

        #     print("Cutting the rope")
        #     rope_creator.get_joints()[5].GetPrim().GetAttribute('physics:jointEnabled').Set(False)

        simulation_app.update()
        time.sleep(0.01)
except KeyboardInterrupt:
    print("Simulation stopped by user")
finally:
    # Stop the simulation
    timeline.stop()
    simulation_app.close()