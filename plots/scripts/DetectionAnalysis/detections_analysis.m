close all; clc;
clearvars -except colors log T_annot

use_manual_annotation = true;

VEHICLE = "serravalle_niguarda";


%% Paths
normal_path = "/home/alessandro/adehome_aida/aida_code/bags";
scriptDir = fileparts(mfilename('fullpath'));
addpath(fullfile(scriptDir,'func'));
addpath(fullfile(scriptDir,'figs'));
graphics_options;

%% Load Data

% load log
if (~exist('log','var'))
    [file,path] = uigetfile(fullfile(normal_path,'*.mat'),'Load log');
    log = load(fullfile(path,file));
end
name1 = 'new';
load_detection_labels;

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

DateTime = datetime(log.t0,'ConvertFrom','epochtime','TicksPerSecond',1e9,'Format','dd-MMM-yyyy HH:mm:ss');

%% NAMING
graphics_options;
col.det.PEDESTRIAN   = colors.blue{2};
col.det.CAR          = colors.green{2};
col.det.TRUCK        = colors.red{2};
col.det.BUS          = colors.purple{2};
col.det.BICYCLE      = colors.yellow{2};
col.det.MOTORCYCLE   = colors.orange{2};
col.det.PED_SIGN     = colors.red{2};
col.det.FP_PED       = colors.red{2};

col.tt.PEDESTRIAN   = colors.blue{1};
col.tt.CAR          = colors.green{1};
col.tt.TRUCK        = colors.red{1};
col.tt.BUS          = colors.purple{1};
col.tt.BICYCLE      = colors.yellow{1};
col.tt.MOTORCYCLE   = colors.orange{1};

load_detection_labels;

sz=3; % Marker size
f=1;
c = 0;
x_lim = [0 inf];

%% PARSING

% URDF Parser loading
urdf_file = fullfile("consts", "vehicle_info",VEHICLE,'vehicle_setup.urdf.xacro');
tree = urdfTree(urdf_file,feval(VEHICLE));

% Vehicle state
ego = load_vehicle_state(log);

VIS_CLASSES = [0,6,7];

% Detection loading
lid_det = load_perception(log);
lid_det = get_det_map_coords(lid_det,ego,tree);

if (use_manual_annotation)
    annot = load_manual_annotations_csv(T_annot);
    % Match timestamps and corresponding class columns (M_annot <= M_lid_det).
    lid_det = update_class(lid_det, annot);
end

ped_det = filter_det_by_class(lid_det, VIS_CLASSES);
ped_det = build_3d_bbox(ped_det);
ped_det.frame_id_rel = lid_det.frame_id_rel;


% Target Tracking
full_tt = load_tt(log);
tt = filter_tt_by_class(full_tt, VIS_CLASSES);

tt_exists = exist('tt', 'var') == 1;

sensors = { ...
    struct('s', ped_det, 'name', 'LD'), ...
};
%% PLOTTING

map;
link_axes();
