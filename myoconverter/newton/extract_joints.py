import xml.etree.ElementTree as ET
import csv
import os
import re

def parse_vec(s):
    if not s:
        return (0.0, 0.0, 0.0)
    return tuple(map(float, s.split()))

def transform_vec(v):
    # Convert from Y-up (OpenSim/rotated) to Z-up (Newton/MuJoCo)
    # Original root has euler="1.571 0 0" (90 deg x-axis rotation)
    # We apply this rotation to all vectors to bake it in.
    # (x, y, z) -> (x, -z, y)
    return (v[0], -v[2], v[1])

def is_path_point_body(name):
    # Heuristic based on names like grac_r_grac_r-P2
    # Check if ends with -P followed by digits
    return bool(re.search(r'-P\d+$', name))

def extract_joints(xml_path, output_csv_path):
    tree = ET.parse(xml_path)
    root = tree.getroot()
    worldbody = root.find('worldbody')
    
    # We want to skip 'ground' and start with 'pelvis' or whatever the main root is.
    # Usually 'pelvis' is a direct child of 'worldbody'.
    
    rows = []
    
    # Header
    header = ['body_name', 'parent_body', 'joint_name', 'joint_type', 'joint_axis', 
              'range_min', 'range_max', 'pos_x', 'pos_y', 'pos_z', 'is_path_point', 'meshes']
    
    # Parse assets
    mesh_map = {} # mesh_name -> filename
    assets = root.find('asset')
    if assets is not None:
        for m in assets.findall('mesh'):
            m_name = m.get('name')
            m_file = m.get('file')
            if m_name and m_file:
                mesh_map[m_name] = m_file

    def process_body(body, parent_name):
        body_name = body.get('name')
        pos_str = body.get('pos', '0 0 0')
        pos = parse_vec(pos_str)
        pos = transform_vec(pos) # Transform position
        
        is_pp = is_path_point_body(body_name)
        
        # Extract meshes
        body_meshes = []
        for g in body.findall('geom'):
            if g.get('type') == 'mesh':
                m_name = g.get('mesh')
                if m_name in mesh_map:
                    body_meshes.append(mesh_map[m_name])
        
        meshes_str = ';'.join(body_meshes)

        joints = body.findall('joint')
        
        if not joints:
            # Fixed joint/body (no DOF)
            # We record it as 'fixed' type
            rows.append({
                'body_name': body_name,
                'parent_body': parent_name,
                'joint_name': '',
                'joint_type': 'fixed',
                'joint_axis': '0 0 1', # Dummy
                'range_min': 0.0,
                'range_max': 0.0,
                'pos_x': pos[0],
                'pos_y': pos[1],
                'pos_z': pos[2],
                'is_path_point': is_pp,
                'meshes': meshes_str
            })
        else:
            for j in joints:
                j_name = j.get('name')
                j_type = j.get('type', 'hinge') # default in MJCF? usually required
                j_axis_str = j.get('axis', '0 0 1')
                j_axis = parse_vec(j_axis_str)
                j_axis = transform_vec(j_axis) # Transform axis
                
                j_range = j.get('range', '0 0')
                r_min, r_max = parse_vec(j_range)[:2] # range usually has 2 values
                
                rows.append({
                    'body_name': body_name,
                    'parent_body': parent_name,
                    'joint_name': j_name,
                    'joint_type': j_type,
                    'joint_axis': f"{j_axis[0]} {j_axis[1]} {j_axis[2]}",
                    'range_min': r_min,
                    'range_max': r_max,
                    'pos_x': pos[0],
                    'pos_y': pos[1],
                    'pos_z': pos[2],
                    'is_path_point': is_pp,
                    'meshes': meshes_str
                })
        
        # Recurse
        for child in body.findall('body'):
            process_body(child, body_name)

    # Find pelvis or just iterate all children of worldbody that are not ground
    for child in worldbody.findall('body'):
        if child.get('name') == 'ground':
            continue
        process_body(child, 'world')
        
    with open(output_csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=header)
        writer.writeheader()
        writer.writerows(rows)
    
    print(f"Extracted {len(rows)} rows to {output_csv_path}")

if __name__ == "__main__":
    model_path = os.path.join(os.path.dirname(__file__), '../models/mjc/Gait2354Simbody/gait2354_cvt1.xml')
    output_path = os.path.join(os.path.dirname(__file__), 'gait2354_kinematics.csv')
    extract_joints(model_path, output_path)
