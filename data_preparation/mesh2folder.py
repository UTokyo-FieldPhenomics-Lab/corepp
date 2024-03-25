import os

if __name__ == "__main__":
    mesh_folder = '/mnt/data/PieterBlok/Potato/Data/3DPotatoTwin/2_SfM/2_pcd'

    for file in os.listdir(mesh_folder):

        if '.' not in file:
            continue

        mesh_id = file.split('.')[0] 

        mesh_subfolder = os.path.join(mesh_folder, mesh_id)

        if not os.path.exists(mesh_subfolder):
            os.mkdir(
                mesh_subfolder
            )

        os.rename(os.path.join(mesh_folder, file), os.path.join(mesh_subfolder, file))
        print(f'{ os.path.join(mesh_folder, file) } -> { os.path.join(mesh_subfolder, file)}')