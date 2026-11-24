close all;
clearvars -except log images_dir images_cam_names T_annot T_timestamps


VIS_CAM = ["left","front","right"];
MATCHED = true;
LID_UNMATCHED = false;
CAM_UNMATCHED = false;
use_manual_annotation               = true;
keep_only_annotated_frames          = true;

VEHICLE = "serravalle_niguarda";


%% Paths
normal_path = "/home/alessandro/hurbie_storage/";
scriptDir = fileparts(mfilename('fullpath'));
addpath(fullfile(scriptDir, '..', '..', 'common'));
addpath(fullfile(scriptDir, '..', '..', 'consts', 'vehicle_info', VEHICLE));
addpath(fullfile(scriptDir, '..', 'shared', 'func'));
addpath(fullfile(scriptDir, 'func'));
addpath(fullfile(scriptDir, 'figs'));

%%
graphics_options;
col.det.PEDESTRIAN   = colors.blue{2};
col.det.CAR          = colors.green{2};
col.det.TRUCK        = colors.red{2};
col.det.BUS          = colors.purple{2};
col.det.BICYCLE      = colors.yellow{2};
col.det.MOTORCYCLE   = colors.orange{2};
col.det.FP_PED       = colors.red{2};

col.det2.PEDESTRIAN   = colors.blue{1};
col.det2.CAR          = colors.green{1};
col.det2.TRUCK        = colors.red{1};
col.det2.BUS          = colors.purple{1};
col.det2.BICYCLE      = colors.yellow{1};
col.det2.MOTORCYCLE   = colors.orange{1};
col.det2.FP_PED       = colors.red{1};



col.tt.PEDESTRIAN   = [1 0 1];
col.tt.CAR          = colors.green{3};
col.tt.TRUCK        = colors.red{3};
col.tt.BUS          = colors.purple{3};
col.tt.BICYCLE      = colors.yellow{3};
col.tt.MOTORCYCLE   = colors.orange{3};

load_detection_labels;

sz=3; % Marker size
f=1;
c = 0;
x_lim = [0 inf];

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

if ~exist('T_timestamps','var')
    [file,path] = uigetfile(fullfile(normal_path,'*.csv'),'Load timestamps.csv');
    if isequal(file, 0)
        return;
    end

    % Read CSV
    filename = fullfile(path, file);
    T_timestamps = readtable(filename, 'Delimiter', ',', ...
        'ReadVariableNames', true, 'VariableNamesLine', 1, 'TextType', 'string');
end

if ~exist('images_dir', 'var') || ~exist('images_cam_names', 'var') || ...
        ~isequal(images_cam_names, cam_names)
    images_cam_names = strings(0);
    images_dir = strings(1, numel(cam_names));
    for i = 1:numel(cam_names)
        image_dir = uigetdir( ...
            normal_path, "Select image " + cam_names(i) + " (" + cam_idx(i) + ") directory");
        if isequal(image_dir, 0)
            return;
        end
        images_dir(i) = string(image_dir);
    end
    images_cam_names = cam_names;
end



%% Parsing

% timestamps
timestamp_fields = cellstr("camera_" + cam_idx);
if ismember('lidar_center', T_timestamps.Properties.VariableNames)
    timestamp_fields{end+1} = 'lidar_center';
end
timestamps = parse_table(T_timestamps, timestamp_fields);

% URDF Parser loading
urdf_file = fullfile(scriptDir, '..', '..', 'consts', 'vehicle_info', VEHICLE, 'vehicle_setup.urdf.xacro');
tree = urdfTree(urdf_file,feval(VEHICLE));

% Vehicle state
ego = load_vehicle_state(log);

% point count
% pt_count = struct('stamp', double(T_counts{:, 1}), 'pt_count', T_counts{:, 2:end});

VIS_CLASSES = [0,6];

% lid_detection loading
[multimod_det, lid_det] = load_perception(log);
cam_det = load_camera_detections(log, VIS_CAM);
multimod_det = get_det_map_coords(multimod_det,ego,tree);
% lid_det = update_struct(lid_det, pt_count, true);

if (use_manual_annotation)
    annot = struct('stamp', double(T_annot{:, 1}), 'class', T_annot{:, 2:end});
    % Match timestamps and corresponding class columns (M_annot <= M_lid_lid_det).
    multimod_det = update_struct(multimod_det, annot, keep_only_annotated_frames);
end

% Extract selected classes
multimod_det = filter_det_by_class(multimod_det, VIS_CLASSES);

% proposal fusion debug 
prop_debug = struct();
for i=1:numel(cam_names)
    prop_debug.(cam_names{i}) = load_proposal_debug(log, cam_names{i});
end
[matched_det, unmatched_det] = get_match_split(prop_debug, multimod_det);


%%
runs = {
    struct('s', matched_det, 'name', 'matched', 'ego', ego, 'col', col.det),...
    struct('s', unmatched_det, 'name', 'UNmatched', 'ego', ego, 'col', col.det2),...
};



%% PLOTTING
map;
% visualize_bbox;
