% Confidence distributions as a function of distance, per VIS_CLASSES.
% One row per sensor and one column per class, with linked XY axes.
% A shared logarithmic color scale shows raw observation counts.
% Populated cells are fully opaque; empty cells are transparent.

class_ids = unique(VIS_CLASSES(:), 'stable');
label_names = fieldnames(DET_LABELS);
label_ids = cellfun(@(name) DET_LABELS.(name), label_names);
[known, label_idx] = ismember(class_ids, label_ids);
if any(~known)
    warning('distance_confidence:UnknownClass', ...
        'Ignoring class IDs absent from DET_LABELS: %s.', mat2str(class_ids(~known).'));
end
class_ids = class_ids(known);
class_names = label_names(label_idx(known));

distance_confidence_figure = figure("Name","DISTANCE-CONFIDENCE",'NumberTitle','off');
distance_confidence_axes = gobjects(numel(sensors), numel(class_ids));

if isempty(sensors) || isempty(class_ids)
    ax_empty = axes('Parent', distance_confidence_figure);
    axis(ax_empty, 'off');
    text(ax_empty, 0.5, 0.5, 'No sensors or classes selected.', ...
        'HorizontalAlignment', 'center');
else

    % Valid sensor detections
    pairs = cell(numel(sensors), numel(class_ids));
    max_distance = 1;

    for i = 1:numel(sensors)
        s = sensors{i}.s;

        % Compute planar distance
        distance = hypot(s.x_cog, s.y_cog);

        valid = isfinite(s.stamp) & isfinite(s.score) & ...
            isfinite(distance) & ...
            s.score >= 0 & s.score <= 1;

        for j = 1:numel(class_ids)
            selected = valid & s.class == class_ids(j);

            distances = double(distance(selected));
            scores = double(s.score(selected));

            pairs{i,j} = [distances(:), scores(:)];
            max_distance = max([max_distance; distances(:)]);
        end
    end

    % Shared bins
    distance_edges = linspace(0, max_distance, 33);
    confidence_edges = linspace(0, 1, 21);

    distance_centers = (distance_edges(1:end-1) + distance_edges(2:end))/2;
    confidence_centers = (confidence_edges(1:end-1) + confidence_edges(2:end))/2;

    layout = tiledlayout(distance_confidence_figure, ...
        numel(sensors), numel(class_ids), ...
        'Padding', 'compact', 'TileSpacing', 'compact');

    max_count = 1;

    % --- Sensor detections ---
    for i = 1:numel(sensors)
        for j = 1:numel(class_ids)
            ax_heatmap = nexttile(layout);
            distance_confidence_axes(i,j) = ax_heatmap;

            data = pairs{i,j};

            counts = histcounts2(data(:,1), data(:,2), ...
                distance_edges, confidence_edges);

            max_count = max([max_count; counts(:)]);

            class_color = col.det.(class_names{j});

            img = imagesc(ax_heatmap, ...
                distance_centers, confidence_centers, counts.');

            colormap(ax_heatmap, turbo(256));

            img.AlphaData = double(counts.' > 0);
            img.AlphaDataMapping = 'none';

            set(ax_heatmap, ...
                'YDir', 'normal', ...
                'Color', [1 1 1], ...
                'XLim', distance_edges([1 end]), ...
                'YLim', [0 1]);

            xlabel(ax_heatmap, 'Distance [m]');
            ylabel(ax_heatmap, 'Confidence');

            hold(ax_heatmap, 'on');

            class_key = patch(ax_heatmap, NaN, NaN, class_color, ...
                'EdgeColor', 'none', ...
                'DisplayName', sprintf('%s - %s (N=%d)', ...
                sensors{i}.name, class_names{j}, size(data,1)));

            legend(ax_heatmap, class_key, ...
                'Location', 'northeast', 'Interpreter', 'none');

            grid(ax_heatmap, 'off');

            if isempty(data)
                text(ax_heatmap, mean(distance_edges([1 end])), 0.5, ...
                    'No valid detections', ...
                    'HorizontalAlignment', 'center');
            end
        end
    end

    % Logarithmic colors keep sparse bins visible beside dense bins.
    set(distance_confidence_axes, ...
        'ColorScale', 'log', ...
        'CLim', [1 max(2, max_count)]);

    cb = colorbar(distance_confidence_axes(end));
    cb.Layout.Tile = 'east';
    cb.Label.String = 'Detection observations per bin (log scale)';
    cb.Label.Interpreter = 'none';

    linkaxes(distance_confidence_axes(:), 'xy');
end