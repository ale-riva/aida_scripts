close all; clc;
clearvars -except colors log

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


DateTime = datetime(log.t0,'ConvertFrom','epochtime','TicksPerSecond',1e9,'Format','dd-MMM-yyyy HH:mm:ss');

%% NAMING
graphics_options;
col.lidar   = colors.green{2};
col.pp      = colors.green{3};
col.radar   = [77 190 238] / 255;
col.conly  = colors.yellow{2};
col.cenh  = colors.orange{2};
col.cenh2   = [1 0 1]; % magenta
col.conly2   = [0.5 0 0.5]; % purple
col.ref     = colors.black;
sz=3; % Marker size
f=1;
c = 0;
x_lim = [0 inf];

%% PARSING

% URDF Parser loading
urdf_file = fullfile("vehicle_info",VEHICLE,'vehicle_setup.urdf.xacro');
tree = urdfTree(urdf_file,feval(VEHICLE));

% Vehicle state
ego = load_vehicle_state(log);

% Detection loading
lid_det = load_perception(log);
lid_det = load_perception_map(lid_det,ego,tree);
ped_det = filter_by_class(lid_det, 0);
ped_det = build_3d_bbox(ped_det);
ped_det.frame_id_rel = lid_det.frame_id_rel;

sensors = { ...
    struct('s', ped_det, 'col', col.lidar, 'name', 'LD - Pedestrian'), ...
};
%% PLOTTING

map;
link_axes();
