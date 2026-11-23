close all;
clearvars -except log images_dir T_annot


VIS_CAM = ["front","left","right"];
use_manual_annotation               = true;
keep_only_annotated_frames          = true;

VEHICLE = "serravalle_niguarda";


%% Paths
normal_path = "/home/alessandro/hurbie_storage/";
scriptDir = fileparts(mfilename('fullpath'));
addpath(fullfile(scriptDir, '..', '..', 'common'));
addpath(fullfile(scriptDir, '..', '..', 'consts', 'vehicle_info', VEHICLE));
addpath(fullfile(scriptDir, 'func'));
addpath(fullfile(scriptDir, 'figs'));

%% Load Data

cam_data = feval(VEHICLE + "_camera_indices");
cam_names = VIS_CAM;
cam_idx = strings(size(cam_names));
for i = 1:numel(cam_names)
    cam_idx(i) = string(cam_data.(cam_names(i)));
end


if ~exist('log', 'var')
    [file, path] = uigetfile(fullfile(normal_path, '*.mat'), 'Load log');
    if isequal(file, 0)
        return;
    end
    log = load(fullfile(path, file));
end

if (use_manual_annotation)
    if ~exist('T_annot','var')
        [file,path] = uigetfile(fullfile(normal_path,'*.csv'),'Load manual annotations');
        if isequal(file, 0)
            return;
        end

        % Read CSV
        filename = fullfile(path, file);
        T_annot = readtable(filename);
    end
end

if ~exist('images_dir', 'var')
    images_dir = strings(1, numel(cam_names));
    for i = 1:numel(cam_names)
        images_dir(i) = string(uigetdir( ...
            normal_path, "Select image " + cam_names(i) + " (" + cam_idx(i) + ") directory"));
    
        if isequal(images_dir(i), 0)
            return;
        end
    end
end



%% Parsing


% URDF Parser loading
urdf_file = fullfile("consts", "vehicle_info",VEHICLE,'vehicle_setup.urdf.xacro');
tree = urdfTree(urdf_file,feval(VEHICLE));

% Vehicle state
ego = load_vehicle_state(log);

% point count
% pt_count = parse_table(T_counts,'pt_count');

VIS_CLASSES = [0,6];

% Detection loading
det = load_perception(log);
det = get_det_map_coords(det,ego,tree);
% det = update_struct(det, pt_count, true);

if (use_manual_annotation)
    annot = parse_table(T_annot,'class');
    % Match timestamps and corresponding class columns (M_annot <= M_lid_det).
    det = update_struct(det, annot, keep_only_annotated_frames);
end

% proposal fusion debug 
for i=1:numel(cam_names)
    prop_debug.(cam_names{i}) = load_proposal_debug(log, cam_names{i});
end


prop_debug = filter_proposal_debug(prop_debug, det);





%% PLOTTING
