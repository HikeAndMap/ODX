import os
import json
import math
import shutil

from modules import log
from modules import io
from modules import system
from modules import types
from modules.shots import get_geojson_shots_from_opensfm
from modules.osfm import OSFMContext
from modules import gsd
from modules.point_cloud import export_info_json
from modules.cropper import Cropper
from modules.orthophoto import get_orthophoto_vars, get_max_memory, generate_png
from modules.memory_monitor import monitor as memory_monitor, gib
from modules.run_timing import timing as run_timing, hms as patch9_hms  # ODX's own hms() below would shadow it
from modules.tiles.tiler import generate_colored_hillshade
from modules.utils import get_raster_stats, np_from_json

def hms(seconds):
    h = seconds // 3600
    m = seconds % 3600 // 60
    s = seconds % 3600 % 60
    if h > 0:
        return '{}h:{}m:{}s'.format(h, m, round(s, 0))
    elif m > 0:
        return '{}m:{}s'.format(m, round(s, 0))
    else:
        return '{}s'.format(round(s, 0))


def generate_point_cloud_stats(input_point_cloud, pc_info_file, rerun=False):
    if not os.path.exists(pc_info_file) or rerun:
        export_info_json(input_point_cloud, pc_info_file)

    if os.path.exists(pc_info_file):
        with open(pc_info_file, 'r') as f:
            return json.loads(f.read())

class ODMReport(types.ODM_Stage):
    def process(self, args, outputs):
        tree = outputs['tree']
        reconstruction = outputs['reconstruction']

        if not os.path.exists(tree.odm_report): system.mkdir_p(tree.odm_report)

        log.INFO("Exporting shots.geojson")

        shots_geojson = os.path.join(tree.odm_report, "shots.geojson")
        if not io.file_exists(shots_geojson) or self.rerun():
            # Extract geographical camera shots
            if reconstruction.is_georeferenced():
                # Check if alignment has been performed (we need to transform our shots if so)
                a_matrix = None
                if io.file_exists(tree.odm_georeferencing_alignment_matrix):
                    with open(tree.odm_georeferencing_alignment_matrix, 'r') as f:
                        a_matrix = np_from_json(f.read())
                        log.INFO("Aligning shots to %s" % a_matrix)

                shots = get_geojson_shots_from_opensfm(tree.opensfm_reconstruction, utm_srs=reconstruction.get_proj_srs(), utm_offset=reconstruction.georef.utm_offset(), a_matrix=a_matrix)
            else:
                # Pseudo geo
                shots = get_geojson_shots_from_opensfm(tree.opensfm_reconstruction, pseudo_geotiff=tree.odm_orthophoto_tif)

            if shots:
                with open(shots_geojson, "w") as fout:
                    fout.write(json.dumps(shots))

                log.INFO("Wrote %s" % shots_geojson)
            else:
                log.WARNING("Cannot extract shots")
        else:
            log.WARNING('Found a valid shots file in: %s' % shots_geojson)

        camera_mappings = os.path.join(tree.odm_report, "camera_mappings.npz")
        if not io.file_exists(camera_mappings) or self.rerun():
            src_cm = os.path.join(tree.opensfm, "camera_mappings.npz")
            if io.file_exists(src_cm):
                shutil.copy(src_cm, camera_mappings)
                log.INFO("Copied %s --> %s" % (src_cm, camera_mappings))
            else:
                log.WARNING("Cannot copy camera mappings")
        else:
            log.WARNING("Found a valid camera mappings file in: %s" % camera_mappings)
        
        
        if args.skip_report:
            # Stop right here
            log.WARNING("Skipping report generation as requested")
            return

        # Augment OpenSfM stats file with our own stats
        odm_stats_json = os.path.join(tree.odm_report, "stats.json")
        octx = OSFMContext(tree.opensfm)
        osfm_stats_json = octx.path("stats", "stats.json")
        codem_stats_json = octx.path("stats", "codem", "registration.json")
        odm_stats = None
        point_cloud_file = None
        views_dimension = None

        if not os.path.exists(odm_stats_json) or self.rerun():
            # [PATCH5] the memory figures of an earlier run of this task, kept for the stages this run does not repeat
            previous_memory = None
            previous_timing = None
            if os.path.exists(odm_stats_json):
                try:
                    with open(odm_stats_json, 'r') as f:
                        previous_stats = json.loads(f.read())
                    previous_memory = previous_stats.get('memory_statistics')
                    previous_timing = previous_stats.get('timing_statistics')
                except Exception:
                    previous_memory = None
                    previous_timing = None

            if os.path.exists(osfm_stats_json):
                with open(osfm_stats_json, 'r') as f:
                    odm_stats = json.loads(f.read())

                # Add point cloud stats
                if os.path.exists(tree.odm_georeferencing_model_laz):
                    point_cloud_file = tree.odm_georeferencing_model_laz
                    views_dimension = "UserData"

                    # pc_info_file should have been generated by cropper
                    pc_info_file = os.path.join(tree.odm_georeferencing, "odm_georeferenced_model.info.json")
                    odm_stats['point_cloud_statistics'] = generate_point_cloud_stats(tree.odm_georeferencing_model_laz, pc_info_file, self.rerun())
                else:
                    ply_pc = os.path.join(tree.odm_filterpoints, "point_cloud.ply")
                    if os.path.exists(ply_pc):
                        point_cloud_file = ply_pc
                        views_dimension = "views"

                        pc_info_file = os.path.join(tree.odm_filterpoints, "point_cloud.info.json")
                        odm_stats['point_cloud_statistics'] = generate_point_cloud_stats(ply_pc, pc_info_file, self.rerun())
                    else:
                        log.WARNING("No point cloud found")

                odm_stats['point_cloud_statistics']['dense'] = not args.fast_orthophoto

                # Add runtime stats
                total_time = (system.now_raw() - outputs['start_time']).total_seconds()
                odm_stats['odm_processing_statistics'] = {
                    'total_time': total_time,
                    'total_time_human': hms(total_time),
                    'average_gsd': gsd.opensfm_reconstruction_average_gsd(octx.recon_file(), use_all_shots=reconstruction.has_gcp()),
                }

                # [PATCH5] Add memory stats
                memory_stats = memory_monitor.statistics(previous_memory)
                if memory_stats:
                    odm_stats['memory_statistics'] = memory_stats
                    log.INFO("[PATCH5] Peak memory of the run: %s RAM, %s swap, %s together (highest in the %s stage; measured from %s)" % (
                        gib(memory_stats['peak_ram_bytes']), gib(memory_stats['peak_swap_bytes']), gib(memory_stats['peak_total_bytes']),
                        memory_stats['peak_total_stage'], memory_stats['source']))

                # [PATCH9] Add timing stats
                timing_stats = run_timing.statistics(previous_timing)
                if timing_stats:
                    odm_stats['timing_statistics'] = timing_stats
                    steps = [st for s in timing_stats['stages'] for st in (s.get('steps') or [])]
                    longest = sorted(steps, key=lambda st: st.get('seconds') or 0, reverse=True)[:3]
                    log.INFO("[PATCH9] Time per stage recorded (%s stages, %s steps, %s measured%s); longest steps: %s" % (
                        len(timing_stats['stages']), len(steps), patch9_hms(timing_stats['total_seconds']),
                        ', with an earlier run of this task' if timing_stats.get('merged_with_earlier_run') else '',
                        ', '.join('%s %s' % (st['name'], patch9_hms(st.get('seconds'))) for st in longest) or 'none'))

                # Add CODEM stats
                if os.path.exists(codem_stats_json):
                    with open(codem_stats_json, 'r') as f:
                        odm_stats['align'] = json.loads(f.read())

                with open(odm_stats_json, 'w') as f:
                    f.write(json.dumps(odm_stats))
            else:
                log.WARNING("Cannot generate report, OpenSfM stats are missing")
        else:
            log.WARNING("Reading existing stats %s" % odm_stats_json)
            with open(odm_stats_json, 'r') as f:
                odm_stats = json.loads(f.read())

        # Generate overlap diagram
        if odm_stats is not None and odm_stats.get('point_cloud_statistics') and point_cloud_file and views_dimension:
            bounds = odm_stats['point_cloud_statistics'].get('stats', {}).get('bbox', {}).get('native', {}).get('bbox')
            if bounds:
                image_target_size = 1400 # pixels
                osfm_stats_dir = os.path.join(tree.opensfm, "stats")
                diagram_tiff = os.path.join(osfm_stats_dir, "overlap.tif")
                diagram_png = os.path.join(osfm_stats_dir, "overlap.png")

                width = bounds.get('maxx') - bounds.get('minx')
                height = bounds.get('maxy') - bounds.get('miny')
                max_dim = max(width, height)
                resolution = float(max_dim) / float(image_target_size)
                radius = resolution * math.sqrt(2)

                # Larger radius for sparse point cloud diagram
                if not odm_stats['point_cloud_statistics']['dense']:
                    radius *= 10

                system.run("pdal translate -i \"{}\" "
                            "-o \"{}\" "
                            "--writer gdal "
                            "--writers.gdal.resolution={} "
                            "--writers.gdal.data_type=uint8_t "
                            "--writers.gdal.dimension={} "
                            "--writers.gdal.output_type=max "
                            "--writers.gdal.radius={} ".format(point_cloud_file, diagram_tiff, 
                                                                    resolution, views_dimension, radius))
                report_assets = os.path.abspath(os.path.join(os.path.dirname(__file__), "../modules/report"))
                overlap_color_map = os.path.join(report_assets, "overlap_color_map.txt")

                bounds_file_path = os.path.join(tree.odm_georeferencing, 'odm_georeferenced_model.bounds.gpkg')
                if (args.crop > 0 or args.boundary) and os.path.isfile(bounds_file_path):
                    Cropper.crop(bounds_file_path, diagram_tiff, get_orthophoto_vars(args), keep_original=False)

                system.run("gdaldem color-relief \"{}\" \"{}\" \"{}\" -of PNG -alpha".format(diagram_tiff, overlap_color_map, diagram_png))

                # Copy assets
                for asset in ["overlap_diagram_legend.png", "dsm_gradient.png"]:
                    shutil.copy(os.path.join(report_assets, asset), os.path.join(osfm_stats_dir, asset))

                # Generate previews of ortho/dsm
                if os.path.isfile(tree.odm_orthophoto_tif):
                    osfm_ortho = os.path.join(osfm_stats_dir, "ortho.png")
                    generate_png(tree.odm_orthophoto_tif, osfm_ortho, image_target_size)
                
                dems = []
                if args.dsm:
                    dems.append("dsm")
                if args.dtm:
                    dems.append("dtm")

                for dem in dems:
                    dem_file = tree.path("odm_dem", "%s.tif" % dem)
                    if os.path.isfile(dem_file):
                        # Resize first (faster)
                        resized_dem_file = io.related_file_path(dem_file, postfix=".preview")
                        system.run("gdal_translate -outsize {} 0 \"{}\" \"{}\" --config GDAL_CACHEMAX {}%".format(image_target_size, dem_file, resized_dem_file, get_max_memory()))

                        log.INFO("Computing raster stats for %s" % resized_dem_file)
                        dem_stats = get_raster_stats(resized_dem_file)
                        if len(dem_stats) > 0:
                            odm_stats[dem + '_statistics'] = dem_stats[0]

                        osfm_dem = os.path.join(osfm_stats_dir, "%s.png" % dem)
                        colored_dem, hillshade_dem, colored_hillshade_dem = generate_colored_hillshade(resized_dem_file)
                        system.run("gdal_translate -outsize {} 0 -of png \"{}\" \"{}\" --config GDAL_CACHEMAX {}%".format(image_target_size, colored_hillshade_dem, osfm_dem, get_max_memory()))
                        for f in [resized_dem_file, colored_dem, hillshade_dem, colored_hillshade_dem]:
                            if os.path.isfile(f):
                                os.remove(f)
            else:
                log.WARNING("Cannot generate overlap diagram, cannot compute point cloud bounds")
        else:
            log.WARNING("Cannot generate overlap diagram, point cloud stats missing")

        if odm_stats is not None:
            octx.export_report(os.path.join(tree.odm_report, "report.pdf"), odm_stats, self.rerun())
