import shutil, os, glob, math, sys

from modules import log
from modules import io
from modules import system
from modules import context
from modules import point_cloud
from modules import types
from modules.gpu import has_gpu
from modules.utils import get_depthmap_resolution
from modules.osfm import OSFMContext
from modules.multispectral import get_primary_band_name
from modules.point_cloud import fast_merge_ply

ROI_WIDEN_XY = 20.0     # [PATCH13] metres added round the sparse points sideways ...
ROI_WIDEN_Z = 60.0      # ... and in height


def with_sparse_roi(scene_file, openmvs_dir, depthmaps_dir, gpu_config):
    """[PATCH13] A copy of scene_file (scene_roi.mvs) with a region of interest: the box of the scene's sparse points
    (OpenMVS --estimate-roi 1), widened by ROI_WIDEN_XY / ROI_WIDEN_Z. Returns its path, or scene_file when no box came out.
    Both runs use the depth-map folder as the working folder, as every other OpenMVS run here does: the saved scene stores its
    photos' paths relative to it (without -w they were stored relative to the current folder, and the split's sub-scenes then
    could not find a single photo)."""
    estimated = os.path.join(openmvs_dir, "scene_roi_estimated.txt")
    roi_file = os.path.join(openmvs_dir, "scene_roi.txt")
    roi_scene = os.path.join(openmvs_dir, "scene_roi.mvs")
    for f in (estimated, roi_file, roi_scene):
        if os.path.exists(f):
            os.remove(f)
    system.run('"%s" "%s" --estimate-roi 1 --export-roi-file "%s" -w "%s" -v 0 %s' % (context.omvs_densify_path, scene_file, estimated, depthmaps_dir, ' '.join(gpu_config)))
    if not os.path.exists(estimated):
        log.WARNING("[PATCH13] OpenMVS found no region of interest for the scene; it is split as it is")
        return scene_file
    with open(estimated) as f:
        v = [float(x) for x in f.read().split()]
    rot, pos, ext = v[:9], v[9:12], v[12:15]
    if rot != [1, 0, 0, 0, 1, 0, 0, 0, 1]:
        log.WARNING("[PATCH13] The region of interest is rotated; the scene is split with it as estimated")
        shutil.copyfile(estimated, roi_file)
    else:
        ext = [ext[0] + ROI_WIDEN_XY, ext[1] + ROI_WIDEN_XY, ext[2] + ROI_WIDEN_Z]
        with open(roi_file, "w") as f:
            f.write("1 0 0\n0 1 0\n0 0 1\n%s\n" % "\n".join("%.6f" % x for x in pos + ext))
    system.run('"%s" "%s" --import-roi-file "%s" -o "%s" -w "%s" --archive-type 3 -v 0 %s' % (context.omvs_densify_path, scene_file, roi_file, roi_scene, depthmaps_dir, ' '.join(gpu_config)))
    if not os.path.exists(roi_scene):
        log.WARNING("[PATCH13] The region of interest could not be set; the scene is split as it is")
        return scene_file
    log.INFO("[PATCH13] Scene split within the region of its sparse points: x %.1f to %.1f, y %.1f to %.1f, height %.1f to %.1f"
             % (pos[0] - ext[0], pos[0] + ext[0], pos[1] - ext[1], pos[1] + ext[1], pos[2] - ext[2], pos[2] + ext[2]))
    return roi_scene


class ODMOpenMVSStage(types.ODM_Stage):
    def process(self, args, outputs):
        # get inputs
        tree = outputs['tree']
        reconstruction = outputs['reconstruction']
        photos = reconstruction.photos
        octx = OSFMContext(tree.opensfm)
        pc_tile = False

        if not photos:
            raise system.ExitException('Not enough photos in photos array to start OpenMVS')

        # check if reconstruction was done before
        if not io.file_exists(tree.openmvs_model) or self.rerun():
            if self.rerun():
                if io.dir_exists(tree.openmvs):
                    shutil.rmtree(tree.openmvs)

            # export reconstruction from opensfm
            openmvs_scene_file = os.path.join(tree.openmvs, "scene.mvs")
            if not io.file_exists(openmvs_scene_file) or self.rerun():
                cmd = 'export_openmvs'
                octx.run(cmd)
            else:
                log.WARNING("Found existing %s" % openmvs_scene_file)
            
            self.update_progress(10)

            depthmaps_dir = os.path.join(tree.openmvs, "depthmaps")

            if io.dir_exists(depthmaps_dir) and self.rerun():
                shutil.rmtree(depthmaps_dir)

            if not io.dir_exists(depthmaps_dir):
                os.mkdir(depthmaps_dir)

            depthmap_resolution = get_depthmap_resolution(args, photos)
            log.INFO("Depthmap resolution set to: %spx" % depthmap_resolution)

            if outputs["undist_image_max_size"] <= depthmap_resolution:
                resolution_level = 0
            else:
                resolution_level = int(round(math.log(outputs['undist_image_max_size'] / float(depthmap_resolution)) / math.log(2)))

            log.INFO("Running dense reconstruction. This might take a while.")
            
            log.INFO("Estimating depthmaps")
            number_views_fuse = 2
            densify_ini_file = os.path.join(tree.openmvs, 'Densify.ini')
            subres_levels = 2 # The number of lower resolutions to process before estimating output resolution depthmap.
            filter_point_th = -20
            min_resolution = 320 if args.pc_quality in ["low", "lowest"] else 640

            config = [
                "--resolution-level %s" % int(resolution_level),
                '--dense-config-file "%s"' % densify_ini_file,
                "--max-resolution %s" % int(outputs['undist_image_max_size']),
                "--min-resolution %s" % min_resolution,
                "--max-threads %s" % args.max_concurrency,
                "--number-views-fuse %s" % number_views_fuse,
                "--sub-resolution-levels %s" % subres_levels,
                "--archive-type 3",
                '-w "%s"' % depthmaps_dir, 
                "-v 0"
            ]

            gpu_config = []
            use_gpu = has_gpu(args)
            if use_gpu:
                gpu_config.append("--cuda-device -1")
            else:
                gpu_config.append("--cuda-device -2")

            extra_config = []

            if args.pc_skip_geometric:
                extra_config.append("--geometric-iters 0")

            masks_dir = os.path.join(tree.opensfm, "undistorted", "masks")
            masks = os.path.exists(masks_dir) and len(os.listdir(masks_dir)) > 0
            if masks:
                extra_config.append("--ignore-mask-label 0")

            with open(densify_ini_file, 'w+') as f:
                f.write("Optimize = 7\nMin Views Filter = 1\n")

            def run_densify():
                system.run('"%s" "%s" %s' % (context.omvs_densify_path, 
                                        openmvs_scene_file,
                                        ' '.join(config + gpu_config + extra_config)))
            
            if os.getenv("ODX_LOW_MEMORY"):
                pc_tile = True
                config.append("--fusion-mode 1")

            # [PATCH16] A task with many photos is made in pieces from the start. In one piece the dense point cloud of some
            # thousands of 50 MP photos needs more than the RAM (Burnham, 9,246 photos: out of memory after 18.5 h), and ODX
            # only cut it into pieces after running out of memory. The pieces share the camera positions of the whole task.
            tile_from = getattr(args, 'pc_tile_from_photos', 0) or 0
            if pc_tile:
                log.INFO("[PATCH16] ODX_LOW_MEMORY is set: the dense point cloud is made in pieces")
            elif tile_from > 0 and len(photos) >= tile_from:
                pc_tile = True
                config.append("--fusion-mode 1")
                log.INFO("[PATCH16] %s photos, at least pc-tile-from-photos (%s): the dense point cloud is made in pieces from the start"
                         % (len(photos), tile_from))
            elif tile_from > 0:
                log.INFO("[PATCH16] %s photos, fewer than pc-tile-from-photos (%s): the dense point cloud is made in one piece"
                         % (len(photos), tile_from))
            else:
                log.INFO("[PATCH16] pc-tile-from-photos is 0: the dense point cloud is tried in one piece first")

            try:
                run_densify()
            except system.SubprocessException as e:
                # If the GPU was enabled and the program failed,
                # try to run it again without GPU
                if e.errorCode == 1 and use_gpu:
                    log.WARNING("OpenMVS failed with GPU, is your graphics card driver up to date? Falling back to CPU.")
                    gpu_config = ["--cuda-device -2"]
                    run_densify()
                elif (e.errorCode == 137 or e.errorCode == 143 or e.errorCode == 3221226505) and not pc_tile:
                    log.WARNING("OpenMVS ran out of memory, we're going to turn on tiling to see if we can process this.")
                    pc_tile = True
                    config.append("--fusion-mode 1")
                    run_densify()
                else:
                    raise e

            self.update_progress(85)
            files_to_remove = []
            scene_dense = os.path.join(tree.openmvs, 'scene_dense.mvs')

            if pc_tile:
                log.INFO("Computing sub-scenes")

                # [PATCH13] OpenMVS 2.2.0 computes the box round a scene without a region of interest (ROI) from all depth-map
                # samples in single precision; at heights of about 1,440 m (Baguio) that box ended at 330 m, every sub-scene's
                # box lay below the ground and the fusion trimmed every point away ("Point-cloud trimmed to ROI", 0 points,
                # "no PLY files available"; Burnham Park, 9,246 photos, 2026-11-18). A scene that has a ROI skips that
                # computation, so the scene gets one first: the box of its own sparse points (--estimate-roi 1), widened.
                split_scene_file = openmvs_scene_file
                try:
                    split_scene_file = with_sparse_roi(openmvs_scene_file, tree.openmvs, depthmaps_dir, gpu_config)
                except Exception as e:
                    log.WARNING("[PATCH13] Cannot set the region of interest before splitting, OpenMVS estimates it: %s" % str(e))

                subscene_densify_ini_file = os.path.join(tree.openmvs, 'subscene-config.ini')
                with open(subscene_densify_ini_file, 'w+') as f:
                    f.write("Optimize = 0\nEstimation Geometric Iters = 0\nMin Views Filter = 1\n")

                config = [
                    "--sub-scene-area 660000", # 8000
                    "--max-threads %s" % args.max_concurrency,
                    '-w "%s"' % depthmaps_dir, 
                    "-v 0",
                ]
                system.run('"%s" "%s" %s' % (context.omvs_densify_path, 
                                        split_scene_file,
                                        ' '.join(config + gpu_config)))
                
                scene_files = glob.glob(os.path.join(tree.openmvs, "scene_[0-9][0-9][0-9][0-9].mvs"))
                if len(scene_files) == 0:
                    raise system.ExitException("No OpenMVS scenes found. This could be a bug, or the reconstruction could not be processed.")

                log.INFO("Fusing depthmaps for %s scenes" % len(scene_files))
                
                scene_ply_files = []

                for sf in scene_files:
                    p, _ = os.path.splitext(sf)
                    scene_ply_unfiltered = p + "_dense.ply"
                    scene_ply = p + "_dense_dense_filtered.ply"
                    scene_dense_mvs = p + "_dense.mvs"

                    files_to_remove += [scene_ply, sf, scene_dense_mvs, scene_ply_unfiltered]
                    scene_ply_files.append(scene_ply)

                    if not io.file_exists(scene_ply) or self.rerun():
                        # Fuse
                        config = [
                            '--resolution-level %s' % int(resolution_level),
                            '--max-resolution %s' % int(outputs['undist_image_max_size']),
                            "--sub-resolution-levels %s" % subres_levels,
                            '--dense-config-file "%s"' % subscene_densify_ini_file,
                            '--number-views-fuse %s' % number_views_fuse,
                            '--max-threads %s' % args.max_concurrency,
                            '--archive-type 3',
                            '--postprocess-dmaps 0',
                            '--geometric-iters 0',
                            '-w "%s"' % depthmaps_dir,
                            '-v 0',
                        ]

                        try:
                            system.run('"%s" "%s" %s' % (context.omvs_densify_path, sf, ' '.join(config + gpu_config + extra_config)))
                        except:
                            log.WARNING("Sub-scene %s could not be reconstructed, skipping..." % sf)

                        if not io.file_exists(scene_ply_unfiltered):
                            scene_ply_files.pop()
                            log.WARNING("Could not compute PLY for subscene %s" % sf)
                        else:
                            # Filter
                            if args.pc_filter > 0:
                                system.run('"%s" "%s" --filter-point-cloud %s -v 0 --archive-type 3 %s' % (context.omvs_densify_path, scene_dense_mvs, filter_point_th, ' '.join(gpu_config)))
                            else:
                                # Just rename
                                log.INFO("Skipped filtering, %s --> %s" % (scene_ply_unfiltered, scene_ply))
                                os.rename(scene_ply_unfiltered, scene_ply)
                    else:
                        log.WARNING("Found existing dense scene file %s" % scene_ply)

                # Merge
                log.INFO("Merging %s scene files" % len(scene_ply_files))
                if len(scene_ply_files) == 0:
                    raise system.ExitException("Could not compute dense point cloud (no PLY files available).")
                if len(scene_ply_files) == 1:
                    # Simply rename
                    os.replace(scene_ply_files[0], tree.openmvs_model)
                    log.INFO("%s --> %s"% (scene_ply_files[0], tree.openmvs_model))
                else:
                    # Merge
                    fast_merge_ply(scene_ply_files, tree.openmvs_model)
            else:
                def skip_filtering():
                    # Just rename
                    scene_dense_ply = os.path.join(tree.openmvs, 'scene_dense.ply')
                    if not os.path.exists(scene_dense_ply):
                        raise system.ExitException("Dense reconstruction failed. This could be due to poor georeferencing or insufficient image overlap.")

                    log.INFO("Skipped filtering, %s --> %s" % (scene_dense_ply, tree.openmvs_model))
                    os.rename(scene_dense_ply, tree.openmvs_model)

                # Filter all at once
                if args.pc_filter > 0:
                    if os.path.exists(scene_dense):
                        config = [
                            "--filter-point-cloud %s" % filter_point_th,
                            '-i "%s"' % scene_dense,
                            "-v 0"
                        ]
                        try:
                            system.run('"%s" %s' % (context.omvs_densify_path, ' '.join(config + gpu_config + extra_config)))
                        except system.SubprocessException as e:
                            if e.errorCode == 137 or e.errorCode == 143 or e.errorCode == 3221226505:
                                log.WARNING("OpenMVS filtering ran out of memory, visibility checks will be skipped.")
                                skip_filtering()
                            else:
                                raise e
                    else:
                        raise system.ExitException("Cannot find scene_dense.mvs, dense reconstruction probably failed. Exiting...")
                else:
                    skip_filtering()

            self.update_progress(95)

            if args.optimize_disk_space:
                files = [scene_dense,
                         os.path.join(tree.openmvs, 'scene_dense.ply'),
                         os.path.join(tree.openmvs, 'scene_dense_dense_filtered.mvs'),
                         octx.path("undistorted", "tracks.csv"),
                         octx.path("undistorted", "reconstruction.json")
                        ] + files_to_remove
                for f in files:
                    if os.path.exists(f):
                        os.remove(f)
                shutil.rmtree(depthmaps_dir)
        else:
            log.WARNING('Found a valid OpenMVS reconstruction file in: %s' %
                            tree.openmvs_model)
