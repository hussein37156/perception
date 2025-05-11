% Copyright (c) 2017 Jacobs University Robotics Group
% All rights reserved.
%
% Unless specified otherwise this code examples are released under 
% Creative Commons CC BY-NC-ND 4.0 license (free for non-commercial use). 
% Details may be found here: https://creativecommons.org/licenses/by-nc-nd/4.0/
%
% If you are interested in using this code commercially, 
% please contact us.
%
% THIS SOFTWARE IS PROVIDED BY Jacobs Robotics ``AS IS'' AND ANY
% EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
% WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
% DISCLAIMED. IN NO EVENT SHALL Jacobs Robotics BE LIABLE FOR ANY
% DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
% (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
% LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
% ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
% (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
% SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
%
% Contact: robotics@jacobs-university.de

clear;
clc;

% image size
width = 1280/2;
height = 720/2;
ImgSize = [height,width];

% camera and lens parameters
K=[700.515/2 0 662.935/2;
   0 700.515/2 353.9215/2;
   0 0 1]; %camera intrinsics
% distortion parameters
radialDist = [0,0];
tangentialDist = [0,0];

% setup parameters
d1 = 6; % glass thickness [mm]
d0 = 2; % distance of camera lens from inside of glass [mm]
d0virtual = [0;0;0.5382]; % virtual d0 distance (found by running Find_Optimal_d0)
ng = 1.492; % glass refraction index
nw = 1.345; % water refraction index

d2=d0+d1;
mu_v=[ng;nw] ;
n=[0;0;1];

% image points
ImgPts=zeros(3,width*height);
N=size(ImgPts,2);
for i=1:width
    for j=1:height
        ImgPts(1,(j-1)*width+i)=i;
        ImgPts(2,(j-1)*width+i)=j;
        ImgPts(3,(j-1)*width+i)=1;
    end    
end

Rays = inv(K)*ImgPts;
M = zeros(3,N);
for i=1:N
    (i/N)*100
    p=5000*Rays(:,i)+d0virtual;    
    M(:,i) = SolveForwardProjectionCase3(d0,d2,-n,mu_v,p);
end

worldPoints = M.'; % transpose M (3xN -> Nx3)

rvec = [0 0 0]; % rotation 
tvec = [0 0 0]; % translation 
tform = rigidtform3d(rvec,tvec);

focalLength = [K(1,1),K(2,2)];
principalPoint = [K(1,3),K(2,3)];
intrinsics = cameraIntrinsics(focalLength,principalPoint,ImgSize,"RadialDistortion",radialDist,"TangentialDistortion",tangentialDist);

[imagePoints, validIndexes] = world2img(worldPoints, tform, intrinsics, ApplyDistortion=true);

Mx = imagePoints(:,1); % first col of imagePoints (Nx2 matrix)
Mx = Mx.'; % transpose to make a 1xN matrix
My = imagePoints(:,2); % second col of imagePoints (Nx2 matrix)
My = My.';

mapx=zeros(height,width); 
mapy=zeros(height,width);

for i=1:width
    for j=1:height
        mapx(j,i)=Mx((j-1)*width+i);
        mapy(j,i)=My((j-1)*width+i);
    end    
end

% export mapx and mapy into a .txt file with comma separation
writematrix(mapx,"MapX.txt")
writematrix(mapy,"MapY.txt")

fprintf("Process complete.")
