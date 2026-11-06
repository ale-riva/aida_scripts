
figure("Name","MAP",'NumberTitle','off');
% MAP
tiledlayout(1,2,'Padding','compact','TileSpacing','compact')
confidence_map_axes = nexttile;

% confidence map
hold on; axis equal;
hold on; axis equal;
xlabel("X [m]")
ylabel("Y [m]")

plot(ego.x, ego.y, ...
    'LineWidth', 0.3, ...
    'Color', [0.5 0.5 0.5], ...
    'DisplayName', 'Ego');

% Confidence colormap
colormap(turbo);
clim([0 1]);
cb = colorbar;
cb.Label.String = 'Confidence [-]';

class_names = fieldnames(DET_LABELS);

% --- Target tracking ---
if tt_exists
    valid = ...
        isfinite(tt.stamp) & ...
        isfinite(tt.x_map) & ...
        isfinite(tt.y_map);

    for j = 1:numel(class_names)
        class_name = class_names{j};
        tt_class_idx = valid & tt.class == DET_LABELS.(class_name);

        if ~any(tt_class_idx(:))
            continue;
        end

        plot( ...
            tt.x_map(tt_class_idx), ...
            tt.y_map(tt_class_idx), ...
            'o', ...
            'MarkerSize', 6, ...
            'Color', col.tt.(class_name), ...
            'DisplayName', sprintf('TT - %s', class_name));
    end
end

% --- Sensor detections ---
for i = 1:numel(sensors)
    s = sensors{i}.s;

    valid = ...
        isfinite(s.stamp) & ...
        isfinite(s.x_map) & ...
        isfinite(s.y_map) & ...
        isfinite(s.score);

    for j = 1:numel(class_names)
        class_name = class_names{j};
        class_idx = valid & s.class == DET_LABELS.(class_name);

        if ~any(class_idx(:))
            continue;
        end

        % Marker based on class
        if strcmp(class_name, 'PEDESTRIAN')
            marker = 'o';
        elseif strcmp(class_name, 'FP_PED')
            marker = 'x';
        else
            marker = '.';
        end

        % Fill only supported markers
        fill_args = {};
        if strcmp(marker, 'o')
            fill_args = {'filled'};
        end

        scatter( ...
            s.x_map(class_idx), ...
            s.y_map(class_idx), ...
            50, ...
            s.score(class_idx), ...
            marker, ...
            fill_args{:}, ...
            'LineWidth', 1, ...
            'DisplayName', sprintf('%s - %s', ...
            sensors{i}.name, class_name));
    end
end

legend show



map_axes= nexttile;
hold on; axis equal;
xlabel("X [m]")
ylabel("Y [m]")
plot(ego.x,ego.y,'LineWidth',0.3,'Color',[0.5 0.5 0.5],'DisplayName','Ego')
t1_start = 0;
class_names = fieldnames(DET_LABELS);


L = 0.1;
% target tracking
if(tt_exists)
    valid = ...
        isfinite(tt.stamp) & ...
        isfinite(tt.x_map) & ...
        isfinite(tt.y_map);
    for j = 1:numel(class_names)
        class_name = class_names{j};
        tt_class_idx = valid & tt.class == DET_LABELS.(class_name);
        if ~any(tt_class_idx(:))
            continue;
        end
        plot(tt.x_map(tt_class_idx), ...
            tt.y_map(tt_class_idx), ...
            'o', ...
            'MarkerSize', 6, ...
            'Color', col.tt.(class_name), ...
            'DisplayName', sprintf('TT - %s',class_name));

        % quiver( ...
        %     tt.x_map(tt_class_idx), ...
        %     tt.y_map(tt_class_idx), ...
        %     L*cos(tt.yaw_map(tt_class_idx)), ...
        %     L*sin(tt.yaw_map(tt_class_idx)), ...
        %     0, ...
        %     'Color', col.tt.(class_name), ...
        %     'HandleVisibility','off', ...
        %     'LineWidth', 2.5);
        % plot_tt(tt.x_map(:,1:tt.max_tracks),tt.y_map(:,1:tt.max_tracks),1,"r",class_name);
    end
end


 % --- sensor detections ---
for i = 1:numel(sensors)
    s = sensors{i}.s;

    valid = ...
        isfinite(s.stamp) & ...
        isfinite(s.x_map) & ...
        isfinite(s.y_map);

    for j = 1:numel(class_names)
        class_name = class_names{j};
        class_idx = valid & s.class == DET_LABELS.(class_name);
        if ~any(class_idx(:))
            continue;
        end

        plot( ...
            s.x_map(class_idx), ...
            s.y_map(class_idx), ...
            '.', ...
            'MarkerSize', 20, ...
            'Color', col.det.(class_name), ...
            'DisplayName', sprintf('%s - %s',sensors{i}.name,class_name));



    end
end
legend show
linkaxes([map_axes, confidence_map_axes], 'xy');
