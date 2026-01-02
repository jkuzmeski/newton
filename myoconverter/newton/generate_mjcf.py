import csv
import xml.etree.ElementTree as ET
from xml.dom import minidom
import os

def prettify(elem):
    """Return a pretty-printed XML string for the Element."""
    rough_string = ET.tostring(elem, 'utf-8')
    reparsed = minidom.parseString(rough_string)
    return reparsed.toprettyxml(indent="  ")

def generate_mjcf(csv_path, output_path):
    # Inertial properties from OpenSim Gait2354 model
    inertial_data = {
        'torso': {'mass': 34.2366, 'inertia': [1.4745, 0.7555, 1.4314]},
        'pelvis': {'mass': 11.777, 'inertia': [0.1028, 0.0871, 0.0579]},
        'femur_r': {'mass': 9.3014, 'inertia': [0.1339, 0.0351, 0.1412]},
        'femur_l': {'mass': 9.3014, 'inertia': [0.1339, 0.0351, 0.1412]},
        'tibia_r': {'mass': 3.7075, 'inertia': [0.0504, 0.0051, 0.0511]},
        'tibia_l': {'mass': 3.7075, 'inertia': [0.0504, 0.0051, 0.0511]},
        'patella_r': {'mass': 0.0862, 'inertia': [0.00000287, 0.00001311, 0.00001311]},
        'patella_l': {'mass': 0.0862, 'inertia': [0.00000287, 0.00001311, 0.00001311]},
        'talus_r': {'mass': 0.1000, 'inertia': [0.0010, 0.0010, 0.0010]},
        'talus_l': {'mass': 0.1000, 'inertia': [0.0010, 0.0010, 0.0010]},
        'calcn_r': {'mass': 1.250, 'inertia': [0.0014, 0.0039, 0.0041]},
        'calcn_l': {'mass': 1.250, 'inertia': [0.0014, 0.0039, 0.0041]},
        'toes_r': {'mass': 0.2166, 'inertia': [0.0001, 0.0002, 0.0010]},
        'toes_l': {'mass': 0.2166, 'inertia': [0.0001, 0.0002, 0.0010]},
    }
    
    # Read CSV
    bodies = {} # name -> {'parent': p, 'pos': pos, 'joints': [], 'is_path_point': bool, 'meshes': []}
    all_meshes = set() # Store all unique mesh filenames
    
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            b_name = row['body_name']
            
            # Parse meshes
            mesh_list = []
            if row.get('meshes'):
                mesh_list = row['meshes'].split(';')
                for m in mesh_list:
                    all_meshes.add(m)
            
            if b_name not in bodies:
                bodies[b_name] = {
                    'parent': row['parent_body'],
                    'pos': f"{row['pos_x']} {row['pos_y']} {row['pos_z']}",
                    'joints': [],
                    'is_path_point': row['is_path_point'] == 'True',
                    'meshes': mesh_list
                }
            
            # Add joint if it exists (not fixed/empty)
            if row['joint_type'] and row['joint_type'] != 'fixed':
                bodies[b_name]['joints'].append({
                    'name': row['joint_name'],
                    'type': row['joint_type'],
                    'axis': row['joint_axis'],
                    'range': f"{row['range_min']} {row['range_max']}"
                })

    # Build hierarchy
    # We need to find children for each body
    children_map = {} # parent_name -> [child_names]
    roots = []
    
    for b_name, data in bodies.items():
        p_name = data['parent']
        if p_name == 'world':
            roots.append(b_name)
        else:
            if p_name not in children_map:
                children_map[p_name] = []
            children_map[p_name].append(b_name)

    # XML Root
    root = ET.Element('mujoco', {'model': 'gait2354_newton'})
    
    # Compiler
    compiler = ET.SubElement(root, 'compiler', {
        'angle': 'radian',
        'inertiafromgeom': 'auto',
        'boundmass': '0.01',  # Increased minimum mass
        'boundinertia': '0.01',  # Increased minimum inertia
        'discardvisual': 'false'
    })
    
    # Defaults
    default = ET.SubElement(root, 'default')
    ET.SubElement(default, 'joint', {
        'limited': 'true',
        'damping': '0.05',
        'armature': '0.00001',
        'stiffness': '0'
    })
    ET.SubElement(default, 'geom', {
        'rgba': '0.8 0.6 .4 1',
        'margin': '0.001',
        'density': '1000'  # Set reasonable density (kg/m^3, similar to water/soft tissue)
    })
    
    # Assets
    asset = ET.SubElement(root, 'asset')
    mesh_name_map = {}
    base_asset_path = "../models/mjc/Gait2354Simbody/"
    
    for m_file in sorted(list(all_meshes)):
        # Generate a safe mesh name from filename
        # e.g. Geometry/sacrum.stl -> sacrum
        base_name = os.path.splitext(os.path.basename(m_file))[0]
        # Ensure unique name if duplicates exist (though set() handles exact paths)
        m_name = base_name 
        mesh_name_map[m_file] = m_name
        
        ET.SubElement(asset, 'mesh', {
            'name': m_name,
            'file': base_asset_path + m_file,
            'scale': '1 1 1'
        })

    # Worldbody
    worldbody = ET.SubElement(root, 'worldbody')
    
    # No ground plane here - viewer will add it
    
    def add_body_recursive(parent_elem, body_name):
        data = bodies[body_name]
        body_elem = ET.SubElement(parent_elem, 'body', {
            'name': body_name,
            'pos': data['pos']
        })
        
        # Add inertial properties if available
        inertia_key = body_name.lower()
        if inertia_key in inertial_data:
            inertia_info = inertial_data[inertia_key]
            ixx, iyy, izz = inertia_info['inertia']
            ET.SubElement(body_elem, 'inertial', {
                'pos': '0 0 0',
                'mass': str(inertia_info['mass']),
                'diaginertia': f"{ixx} {iyy} {izz}"
            })
        
        # Add joints
        for j in data['joints']:
            ET.SubElement(body_elem, 'joint', {
                'name': j['name'],
                'type': j['type'],
                'axis': j['axis'],
                'range': j['range']
            })
            
        # Add joint marker (Red sphere)
        if data['joints']:
             ET.SubElement(body_elem, 'geom', {
                'name': f"{body_name}_joint_marker",
                'type': 'sphere',
                'size': '0.015',
                'rgba': '1 0 0 1', # Red
                'group': '1',
                'contype': '0',
                'conaffinity': '0'
            })
            
        # Add mesh geometry for hydroelastic contacts
        if data['meshes']:
            for i, m_file in enumerate(data['meshes']):
                m_name = mesh_name_map.get(m_file)
                if m_name:
                    # Collision mesh for hydroelastic contacts
                    ET.SubElement(body_elem, 'geom', {
                        'type': 'mesh',
                        'mesh': m_name,
                        'euler': '1.571 0 0', # Rotate 90 deg X to align Y-up mesh to Z-up body
                        'rgba': '0.8 0.6 0.4 1',
                        'contype': '1',
                        'conaffinity': '1'
                    })
                    # Visual mesh (same mesh, visual only)
                    ET.SubElement(body_elem, 'geom', {
                        'type': 'mesh',
                        'mesh': m_name,
                        'euler': '1.571 0 0',
                        'rgba': '0.8 0.6 0.4 1',
                        'group': '1',  # Visual only
                        'contype': '0',
                        'conaffinity': '0'
                    })
        elif data['is_path_point']:
             # Path points - visual only marker
             ET.SubElement(body_elem, 'geom', {
                'type': 'sphere',
                'size': '0.005',
                'rgba': '0 0 1 0.5',
                'group': '1',
                'contype': '0',
                'conaffinity': '0'
            })
        else:
             # Bodies without meshes need collision geometry too!
             # Use a small capsule for collision
             ET.SubElement(body_elem, 'geom', {
                'type': 'capsule',
                'size': '0.04 0.08',  # radius, half-length
                'rgba': '0.8 0.6 0.4 1',
                'contype': '1',  # Enable collision
                'conaffinity': '1'
            })
            
        # Recurse
        if body_name in children_map:
            for child_name in children_map[body_name]:
                add_body_recursive(body_elem, child_name)

    for r in roots:
        add_body_recursive(worldbody, r)
        
    # Contact exclusions (prevent parent-child collisions)
    contact = ET.SubElement(root, 'contact')
    for b_name, data in bodies.items():
        p_name = data['parent']
        if p_name != 'world':
            ET.SubElement(contact, 'exclude', {
                'body1': p_name,
                'body2': b_name
            })

    # Write to file
    xml_str = prettify(root)
    with open(output_path, 'w') as f:
        f.write(xml_str)
    
    print(f"Generated MJCF at {output_path}")

if __name__ == "__main__":
    csv_in = os.path.join(os.path.dirname(__file__), 'gait2354_kinematics.csv')
    xml_out = os.path.join(os.path.dirname(__file__), 'gait2354_newton.xml')
    generate_mjcf(csv_in, xml_out)
