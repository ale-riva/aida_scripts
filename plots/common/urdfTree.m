classdef urdfTree < handle
    %URDFTREE Static frame tree loaded from a URDF or a simple xacro file.
    %   tree = urdfTree(filename, xacroArgs);
    %   tree.parse(filename, xacroArgs);
    %   [x,y,z,roll,pitch,yaw] = tree.transform( ...
    %       x,y,z,roll,pitch,yaw,sourceFrame,targetFrame);
    %   bbox = tree.transformBBox(det_struct.bbox,sourceFrame,targetFrame);
    %
    %   xacroArgs is a struct overriding xacro:arg defaults. The reader
    %   supports xacro:arg, xacro:property, $(arg name), and ${property}.
    %   Expand macros/includes with xacro before loading more complex files.
    %   Only fixed joints contribute transforms. URDF origin rpy is in
    %   radians; public pose angles are in DEGREES, as in build_3d_bbox.
    %   Orientations use Rz(yaw)*Ry(pitch)*Rx(roll). Output pitch is in
    %   [-90,90]; at gimbal lock, output roll is zero.
    %   RootFrame is the unique root, or an empty string for a forest.
    %   No ROS or Robotics System Toolbox is required.

    properties (SetAccess = private)
        Frames = strings(0,1)
        RootFrame = ""
    end

    properties (Access = private)
        Parents = strings(0,1)
        Children = strings(0,1)
        Matrices = zeros(4,4,0) % Each matrix maps child coordinates to parent.
    end

    methods
        function obj = urdfTree(filename,varargin)
            if nargin > 0
                obj.parse(filename,varargin{:});
            end
        end

        function parse(obj,filename,xacroArgs)
            %PARSE Load fixed joints, preserving the old tree on failure.
            if nargin < 3
                xacroArgs = struct;
            end
            validateattributes(xacroArgs,{'struct'},{'scalar'},'parse','xacroArgs');
            parsed = urdfTree;
            parsed.readXacro(filename,xacroArgs);
            if isempty(parsed.Frames)
                error('urdfTree:EmptyTree','The file contains no links or fixed joints.');
            end
            for k = 1:numel(parsed.Frames)
                parsed.toRoot(parsed.Frames(k));
            end
            roots = setdiff(parsed.Frames,parsed.Children,'stable');
            if isscalar(roots)
                parsed.RootFrame = roots;
            end
            obj.Parents = parsed.Parents;
            obj.Children = parsed.Children;
            obj.Matrices = parsed.Matrices;
            obj.Frames = parsed.Frames;
            obj.RootFrame = parsed.RootFrame;
        end

        function T = getTransform(obj,source,target)
            %GETTRANSFORM Homogeneous matrix mapping source into target.
            T = obj.between(source,target);
        end

        function [x,y,z,roll,pitch,yaw] = transform(obj,x,y,z,roll,pitch,yaw,source,target)
            %TRANSFORM Express a pose in target instead of source.
            %   Inputs may be scalars or equally sized real numeric arrays.
            %   Scalars expand to match the arrays; output shapes are retained.
            T = obj.between(source,target);
            values = {x,y,z,roll,pitch,yaw};
            outputSize = [1 1];
            for k = 1:6
                validateattributes(values{k},{'numeric'},{'real'},'transform');
                if ~isscalar(values{k})
                    if ~isequal(outputSize,[1 1]) && ~isequal(size(values{k}),outputSize)
                        error('urdfTree:PoseSize','Pose arrays must have the same size.');
                    end
                    outputSize = size(values{k});
                end
            end
            for k = 1:6
                values{k} = double(values{k}) + zeros(outputSize);
            end
            points = T(1:3,1:3)*[values{1}(:).';values{2}(:).';values{3}(:).'] + T(1:3,4);
            x = reshape(points(1,:),outputSize);
            y = reshape(points(2,:),outputSize);
            z = reshape(points(3,:),outputSize);
            angles = nan(3,numel(x));
            for k = 1:numel(x)
                rpy = [values{4}(k),values{5}(k),values{6}(k)];
                if all(isfinite(rpy))
                    R = T(1:3,1:3)*urdfTree.rpyRotation(rpy);
                    angles(:,k) = urdfTree.rotationRPY(R);
                end
            end
            roll = reshape(angles(1,:),outputSize);
            pitch = reshape(angles(2,:),outputSize);
            yaw = reshape(angles(3,:),outputSize);
        end

        function bbox = transformBBox(obj,bbox,source,target)
            %TRANSFORMBBOX Convert bbox.x/y/z vertices between two frames.
            %   Accepts det_struct.bbox from build_3d_bbox. Dimensions
            %   [N_messages,N_detections,N_vertices] and NaNs are preserved.
            validateattributes(bbox,{'struct'},{'scalar'},'transformBBox','bbox');
            fields = {'x','y','z'};
            for k = 1:3
                if ~isfield(bbox,fields{k})
                    error('urdfTree:InvalidBBox','bbox must contain x, y and z arrays.');
                end
                validateattributes(bbox.(fields{k}),{'numeric'},{'real'},'transformBBox');
            end
            if ~isequal(size(bbox.x),size(bbox.y),size(bbox.z))
                error('urdfTree:BBoxSize','bbox.x, bbox.y and bbox.z must have the same size.');
            end
            T = obj.between(source,target);
            outputSize = size(bbox.x);
            points = T(1:3,1:3)*double([bbox.x(:).';bbox.y(:).';bbox.z(:).']) + T(1:3,4);
            bbox.x = reshape(points(1,:),outputSize);
            bbox.y = reshape(points(2,:),outputSize);
            bbox.z = reshape(points(3,:),outputSize);
        end
    end

    methods (Access = private)
        function readXacro(obj,filename,xacroArgs)
            if ~(ischar(filename) || (isstring(filename) && isscalar(filename))) || ~isfile(filename)
                error('urdfTree:InvalidFile','Provide an existing URDF or xacro filename.');
            end
            document = xmlread(char(filename));
            root = document.getDocumentElement();
            if ~strcmp(char(root.getTagName()),'robot')
                error('urdfTree:InvalidFile','The document root must be a robot element.');
            end
            elements = root.getElementsByTagName('*');
            for k = 0:elements.getLength()-1
                tag = char(elements.item(k).getTagName());
                if startsWith(tag,'xacro:') && ~any(strcmp(tag,{'xacro:arg','xacro:property'}))
                    error('urdfTree:UnsupportedXacro', ...
                        'Unsupported element %s. Expand this file with xacro first.',tag);
                end
            end

            args = containers.Map('KeyType','char','ValueType','char');
            properties = containers.Map('KeyType','char','ValueType','char');
            nodes = root.getElementsByTagName('xacro:arg');
            for k = 0:nodes.getLength()-1
                node = nodes.item(k);
                name = char(node.getAttribute('name'));
                value = char(node.getAttribute('default'));
                if isfield(xacroArgs,name)
                    value = char(string(xacroArgs.(name)));
                end
                args(name) = value;
            end
            nodes = root.getElementsByTagName('xacro:property');
            for k = 0:nodes.getLength()-1
                node = nodes.item(k);
                properties(char(node.getAttribute('name'))) = char(node.getAttribute('value'));
            end

            nodes = root.getElementsByTagName('link');
            for k = 0:nodes.getLength()-1
                name = urdfTree.resolveText(char(nodes.item(k).getAttribute('name')),args,properties);
                obj.Frames(end+1,1) = urdfTree.frameName(name);
            end
            nodes = root.getElementsByTagName('joint');
            for k = 0:nodes.getLength()-1
                node = nodes.item(k);
                if ~strcmp(char(node.getAttribute('type')),'fixed')
                    continue;
                end
                parent = urdfTree.childAttribute(node,'parent','link','');
                child = urdfTree.childAttribute(node,'child','link','');
                parent = urdfTree.resolveText(parent,args,properties);
                child = urdfTree.resolveText(child,args,properties);
                xyz = urdfTree.childAttribute(node,'origin','xyz','0 0 0');
                rpy = urdfTree.childAttribute(node,'origin','rpy','0 0 0');
                xyz = urdfTree.numbers(urdfTree.resolveText(xyz,args,properties));
                rpy = urdfTree.numbers(urdfTree.resolveText(rpy,args,properties));
                obj.addEdge(parent,child,xyz,rad2deg(rpy));
            end
            obj.Frames = unique(obj.Frames,'stable');
        end

        function addEdge(obj,parent,child,t,rpy)
            parent = urdfTree.frameName(parent);
            child = urdfTree.frameName(child);
            R = urdfTree.rpyRotation(rpy);
            T = [R,t(:);0 0 0 1];
            idx = find(obj.Children == child,1);
            if ~isempty(idx)
                if obj.Parents(idx) ~= parent || norm(obj.Matrices(:,:,idx)-T,'fro') > 1e-9
                    error('urdfTree:ConflictingTransform','Conflicting static transform for frame %s.',child);
                end
                return;
            end
            obj.Parents(end+1,1) = parent;
            obj.Children(end+1,1) = child;
            obj.Matrices(:,:,end+1) = T;
            obj.Frames = unique([obj.Frames;parent;child],'stable');
        end

        function T = between(obj,source,target)
            source = urdfTree.frameName(source);
            target = urdfTree.frameName(target);
            for frame = [source,target]
                if ~any(obj.Frames == frame)
                    error('urdfTree:UnknownFrame','Unknown frame: %s.',frame);
                end
            end
            [Ts,sourceRoot] = obj.toRoot(source);
            [Tt,targetRoot] = obj.toRoot(target);
            if sourceRoot ~= targetRoot
                error('urdfTree:DisconnectedFrames','Frames %s and %s have no connecting path.',source,target);
            end
            R = Tt(1:3,1:3).';
            T = [R*Ts(1:3,1:3),R*(Ts(1:3,4)-Tt(1:3,4));0 0 0 1];
        end

        function [T,root] = toRoot(obj,frame)
            T = eye(4);
            root = frame;
            visited = strings(numel(obj.Frames),1);
            nVisited = 0;
            while true
                if any(visited == root)
                    error('urdfTree:Cycle','Cycle in static frame tree at frame %s.',root);
                end
                nVisited = nVisited + 1;
                visited(nVisited) = root;
                idx = find(obj.Children == root,1);
                if isempty(idx)
                    return;
                end
                T = obj.Matrices(:,:,idx)*T;
                root = obj.Parents(idx);
            end
        end
    end

    methods (Static, Access = private)
        function text = resolveText(text,args,properties)
            for k = 1:100
                [match,token] = regexp(text,'\$\(\s*arg\s+([^\)]+)\)','match','tokens','once');
                if ~isempty(match)
                    name = strtrim(token{1});
                    if ~isKey(args,name) || isempty(args(name))
                        error('urdfTree:MissingArgument','Supply xacroArgs.%s for this file.',name);
                    end
                    text = strrep(text,match,args(name));
                    continue;
                end
                [match,token] = regexp(text,'\$\{([^}]+)\}','match','tokens','once');
                if ~isempty(match)
                    name = strtrim(token{1});
                    if ~isKey(properties,name)
                        error('urdfTree:UnsupportedXacro','Unknown property or unsupported expression: %s.',name);
                    end
                    text = strrep(text,match,properties(name));
                    continue;
                end
                if contains(text,'$(')
                    error('urdfTree:UnsupportedXacro','Unsupported substitution: %s.',text);
                end
                return;
            end
            error('urdfTree:XacroCycle','Cyclic or excessively nested xacro substitutions.');
        end

        function value = childAttribute(node,tag,attribute,default)
            children = node.getElementsByTagName(tag);
            value = default;
            if children.getLength() > 0
                child = children.item(0);
                if child.hasAttribute(attribute)
                    value = char(child.getAttribute(attribute));
                end
            end
        end

        function values = numbers(text)
            values = str2double(strsplit(strtrim(text)));
            if numel(values) ~= 3 || any(~isfinite(values))
                error('urdfTree:InvalidOrigin','Origin xyz and rpy must each contain three finite numbers.');
            end
        end

        function name = frameName(value)
            name = regexprep(string(value),'^/+','');
            if ~isscalar(name) || ismissing(name) || strlength(name) == 0
                error('urdfTree:InvalidFrame','Frame IDs must be nonempty scalar names.');
            end
        end

        function R = rpyRotation(rpy)
            c = cosd(rpy); s = sind(rpy);
            R = [c(3)*c(2),c(3)*s(2)*s(1)-s(3)*c(1),c(3)*s(2)*c(1)+s(3)*s(1); ...
                 s(3)*c(2),s(3)*s(2)*s(1)+c(3)*c(1),s(3)*s(2)*c(1)-c(3)*s(1); ...
                 -s(2),c(2)*s(1),c(2)*c(1)];
        end

        function rpy = rotationRPY(R)
            cp = hypot(R(1,1),R(2,1));
            pitch = atan2d(-R(3,1),cp);
            if cp > 1e-12
                roll = atan2d(R(3,2),R(3,3));
                yaw = atan2d(R(2,1),R(1,1));
            else
                roll = 0;
                yaw = atan2d(-R(1,2),R(2,2));
            end
            rpy = [roll;pitch;yaw];
        end
    end
end
