close all;
clearvars -except colors log T_annot T_counts log2 T_annot2 T_counts2

compare                     = true;
use_manual_annotation       = true;
keep_only_annotated_frames  = true;

VEHICLE = "serravalle_niguarda";


%% Paths
normal_path = "/home/alessandro/hurbie_storage/";
scriptDir = fileparts(mfilename('fullpath'));
addpath(fullfile(scriptDir, '..', 'shared', 'func'));
addpath(fullfile(scriptDir,'figs'));
graphics_options;

%% Load Data

% load log
if (~exist('log','var'))
    [file,path] = uigetfile(fullfile(normal_path,'*.mat'),'Load log');
    log = load(fullfile(path,file));
end

if (~exist('T_counts','var'))
    [file,path] = uigetfile(fullfile(normal_path,'*.csv'),'Load det counts');
        if isequal(file, 0)
            return;
        end

        % Read CSV
        filename = fullfile(path, file);
        T_counts = readtable(filename);
end

name1 = 'name1';

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

% compare log
if compare
    if (~exist('log2','var'))
        [file,path] = uigetfile(fullfile(normal_path,'*.mat'),'Load log');
        log2 = load(fullfile(path,file));
    end
    
    if (~exist('T_counts2','var'))
        [file,path] = uigetfile(fullfile(normal_path,'*.csv'),'Load det counts2');
        if isequal(file, 0)
            return;
        end
    
        % Read CSV
        filename = fullfile(path, file);
        T_counts2 = readtable(filename);
    end
    
    name2 = 'name2';
    
    if (use_manual_annotation)
        if ~exist('T_annot2','var')
            [file,path] = uigetfile(fullfile(normal_path,'*.csv'),'Load manual annotations2');
            if isequal(file, 0)
                return;
            end
    
            % Read CSV
            filename = fullfile(path, file);
            T_annot2 = readtable(filename);
        end
    end
end

load_detection_labels;
DateTime = datetime(log.t0,'ConvertFrom','epochtime','TicksPerSecond',1e9,'Format','dd-MMM-yyyy HH:mm:ss');

%% NAMING
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



col.tt.PEDESTRIAN   = colors.blue{3};
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

%% PARSING

% URDF Parser loading
urdf_file = fullfile("consts", "vehicle_info",VEHICLE,'vehicle_setup.urdf.xacro');
tree = urdfTree(urdf_file,feval(VEHICLE));

% Vehicle state
ego = load_vehicle_state(log);
if compare; ego2 = load_vehicle_state(log2); end

% point count
pt_count = struct('stamp', double(T_counts{:, 1}), 'pt_count', T_counts{:, 2:end});
if compare
    pt_count2 = struct('stamp', double(T_counts2{:, 1}), 'pt_count', T_counts2{:, 2:end});
end

VIS_CLASSES = [0,6];

% Detection loading
[multimod_det, lid_det] = load_perception(log);
multimod_det = get_det_map_coords(multimod_det,ego,tree);
multimod_det = update_struct(multimod_det, pt_count, true);

if compare
    lid_det2 = load_perception(log2);
    lid_det2 = get_det_map_coords(lid_det2,ego2,tree);
    lid_det2 = update_struct(lid_det2, pt_count2, true);
end

if (use_manual_annotation)
    annot = struct('stamp', double(T_annot{:, 1}), 'class', T_annot{:, 2:end});
    % Match timestamps and corresponding class columns (M_annot <= M_lid_det).
    multimod_det = update_struct(multimod_det, annot, keep_only_annotated_frames);

    if compare
        annot2 = struct('stamp', double(T_annot2{:, 1}), 'class', T_annot2{:, 2:end});
        lid_det2 = update_struct(lid_det2, annot2, keep_only_annotated_frames);
    end
end

% Extract selected classes
ped_det = filter_det_by_class(multimod_det, VIS_CLASSES);
ped_det = build_3d_bbox(ped_det);
ped_det.frame_id_rel = multimod_det.frame_id_rel;

if compare
    ped_det2 = filter_det_by_class(lid_det2, VIS_CLASSES);
    ped_det2 = build_3d_bbox(ped_det2);
    ped_det2.frame_id_rel = lid_det2.frame_id_rel;
end


% Filter Detections
filters = {
    'x_cog','>=0';
    % 'range','<=35';
};

for i=1:size(filters,1)
    ped_det = filter_struct(ped_det,filters{i,1},filters{i,2});

    if compare
        ped_det2 = filter_struct(ped_det2, filters{i,1},filters{i,2});
    end
    
end

% Target Tracking
full_tt = load_tt(log);
tt = filter_tt_by_class(full_tt, VIS_CLASSES);
if compare
    full_tt2 = load_tt(log2);
    tt2 = filter_tt_by_class(full_tt2, VIS_CLASSES);
end
runs = {struct('s', ped_det, 'name', name1, 'ego', ego, 'tt', tt, 'col', col.det)};
if compare
    runs{end+1} = struct('s', ped_det2, 'name', name2, 'ego', ego2, 'tt', tt2, 'col', col.det2);
end
%% PLOTTING

map;
confidence;
ptcount;
ptcount_confidence;
distance_confidence;
distance_histogram;
statistics;
link_axes();
