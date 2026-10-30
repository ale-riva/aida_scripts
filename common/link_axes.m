function link_axes()
    has_axes = evalin('caller', "(exist('axes','var') && ~isempty(axes)) || (exist('ax','var') && ~isempty(ax))");
    if has_axes
        if evalin('caller', "exist('axes','var') && ~isempty(axes)")
            ax_handles = evalin('caller', 'axes');
        else
            ax_handles = evalin('caller', 'ax');
        end

        plot_axes = ax_handles(isgraphics(ax_handles,'axes'));
    
        if ~isempty(plot_axes)
            if evalin('caller', "exist('tt','var')") && evalin('caller', "isfield(tt,'stamp')")
                tt = evalin('caller', 'tt');
                shared_xlim = [min(tt.stamp, [], 'omitnan') max(tt.stamp, [], 'omitnan')];
                if all(isfinite(shared_xlim)) && shared_xlim(1) < shared_xlim(2)
                    set(plot_axes, 'XLim', shared_xlim);
                end
            end

            if evalin('caller', "exist('link_axes_mode','var')")
                link_axes_mode = evalin('caller', 'link_axes_mode');
            else
                link_axes_mode = 'all';
            end

            switch link_axes_mode
                case 'all'
                    linkaxes(plot_axes,'x');
    
                case 'figure'
                    fig_handles = arrayfun(@(ax) ancestor(ax,'figure'), plot_axes);
                    for fig = reshape(unique(fig_handles), 1, [])
                        fig_axes = plot_axes(fig_handles == fig);
                        if numel(fig_axes) > 1
                            linkaxes(fig_axes,'x');
                        end
                    end

                case 'active'
                    fig_handles = arrayfun(@(ax) ancestor(ax,'figure'), plot_axes);
                    for fig = reshape(unique(fig_handles), 1, [])
                        fig_axes = plot_axes(fig_handles == fig);
                        if numel(fig_axes) > 1
                            linkaxes(fig_axes,'x');
                        end

                        listener = addlistener(fig, 'WindowMousePress', ...
                            @(source, ~) sync_active_figure(source, plot_axes));
                        setappdata(fig, 'LinkAxesActiveFigureListener', listener);
                    end
    
                case 'none'
                    % Keep independent axes for fastest plotting.
    
                otherwise
                    warning('Unknown link_axes_mode "%s". Use "none", "figure", "active", or "all".', link_axes_mode);
            end
        end
    end
end

function sync_active_figure(source_figure, plot_axes)
    source_axes = source_figure.CurrentAxes;
    valid_axes = plot_axes(isgraphics(plot_axes, 'axes'));

    if isempty(source_axes) || ~any(source_axes == valid_axes)
        return;
    end

    set(valid_axes, 'XLim', source_axes.XLim);
end
