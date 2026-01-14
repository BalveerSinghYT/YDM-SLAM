/**
 * This file is part of ORB-SLAM3
 *
 * Copyright (C) 2017-2021 Carlos Campos, Richard Elvira, Juan J. Gómez Rodríguez, José M.M. Montiel and Juan D. Tardós, University of Zaragoza.
 * Copyright (C) 2014-2016 Raúl Mur-Artal, José M.M. Montiel and Juan D. Tardós, University of Zaragoza.
 *
 * ORB-SLAM3 is free software: you can redistribute it and/or modify it under the terms of the GNU General Public
 * License as published by the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * ORB-SLAM3 is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even
 * the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License along with ORB-SLAM3.
 * If not, see <http://www.gnu.org/licenses/>.
 */

#include "Tracking.h"

#include "ORBmatcher.h"
#include "FrameDrawer.h"
#include "Converter.h"
#include "G2oTypes.h"
#include "Optimizer.h"
#include "Pinhole.h"
#include "KannalaBrandt8.h"
#include "MLPnPsolver.h"
#include "GeometricTools.h"

#include <opencv2/calib3d/calib3d.hpp>

#include "FlowNetOnnx.h"

#include <iostream>
#include <algorithm>

#include <cstdlib>
#include <iomanip>
#include <sstream>

#include <cerrno>
#include <cstring>
#include <sys/stat.h>
#include <sys/types.h>

#include <mutex>
#include <chrono>

using namespace std;

namespace ORB_SLAM3
{

    Tracking::Tracking(System *pSys, ORBVocabulary *pVoc, FrameDrawer *pFrameDrawer, MapDrawer *pMapDrawer, Atlas *pAtlas, KeyFrameDatabase *pKFDB, const string &strSettingPath, const int sensor, Settings *settings, const string &_nameSeq) : mState(NO_IMAGES_YET), mSensor(sensor), mTrackedFr(0), mbStep(false), mbOnlyTracking(false), mbMapUpdated(false), mbVO(false), mpORBVocabulary(pVoc), mpKeyFrameDB(pKFDB), mbReadyToInitializate(false), mpSystem(pSys), mpViewer(NULL), bStepByStep(false), mpFrameDrawer(pFrameDrawer), mpMapDrawer(pMapDrawer), mpAtlas(pAtlas), mnLastRelocFrameId(0), time_recently_lost(5.0), mnInitialFrameId(0), mbCreatedMap(false), mnFirstFrameId(0), mpCamera2(nullptr), mpLastKeyFrame(static_cast<KeyFrame *>(NULL))
    {
        // Load camera parameters from settings file
        if (settings)
        {
            newParameterLoader(settings);
        }
        else
        {
            cv::FileStorage fSettings(strSettingPath, cv::FileStorage::READ);

            bool b_parse_cam = ParseCamParamFile(fSettings);
            if (!b_parse_cam)
            {
                std::cout << "*Error with the camera parameters in the config file*" << std::endl;
            }

            // Load ORB parameters
            bool b_parse_orb = ParseORBParamFile(fSettings);
            if (!b_parse_orb)
            {
                std::cout << "*Error with the ORB parameters in the config file*" << std::endl;
            }

            bool b_parse_imu = true;
            if (sensor == System::IMU_MONOCULAR || sensor == System::IMU_STEREO || sensor == System::IMU_RGBD)
            {
                b_parse_imu = ParseIMUParamFile(fSettings);
                if (!b_parse_imu)
                {
                    std::cout << "*Error with the IMU parameters in the config file*" << std::endl;
                }

                mnFramesToResetIMU = mMaxFrames;
            }

            if (!b_parse_cam || !b_parse_orb || !b_parse_imu)
            {
                std::cerr << "**ERROR in the config file, the format is not correct**" << std::endl;
                try
                {
                    throw -1;
                }
                catch (exception &e)
                {
                }
            }
        }

        initID = 0;
        lastID = 0;
        mbInitWith3KFs = false;
        mnNumDataset = 0;

        vector<GeometricCamera *> vpCams = mpAtlas->GetAllCameras();
        std::cout << "There are " << vpCams.size() << " cameras in the atlas" << std::endl;
        for (GeometricCamera *pCam : vpCams)
        {
            std::cout << "Camera " << pCam->GetId();
            if (pCam->GetType() == GeometricCamera::CAM_PINHOLE)
            {
                std::cout << " is pinhole" << std::endl;
            }
            else if (pCam->GetType() == GeometricCamera::CAM_FISHEYE)
            {
                std::cout << " is fisheye" << std::endl;
            }
            else
            {
                std::cout << " is unknown" << std::endl;
            }
        }

#ifdef REGISTER_TIMES
        vdRectStereo_ms.clear();
        vdResizeImage_ms.clear();
        vdORBExtract_ms.clear();
        vdStereoMatch_ms.clear();
        vdIMUInteg_ms.clear();
        vdPosePred_ms.clear();
        vdLMTrack_ms.clear();
        vdNewKF_ms.clear();
        vdTrackTotal_ms.clear();
#endif
    }

#ifdef REGISTER_TIMES
    double calcAverage(vector<double> v_times)
    {
        double accum = 0;
        for (double value : v_times)
        {
            accum += value;
        }

        return accum / v_times.size();
    }

    double calcDeviation(vector<double> v_times, double average)
    {
        double accum = 0;
        for (double value : v_times)
        {
            accum += pow(value - average, 2);
        }
        return sqrt(accum / v_times.size());
    }

    double calcAverage(vector<int> v_values)
    {
        double accum = 0;
        int total = 0;
        for (double value : v_values)
        {
            if (value == 0)
                continue;
            accum += value;
            total++;
        }

        return accum / total;
    }

    double calcDeviation(vector<int> v_values, double average)
    {
        double accum = 0;
        int total = 0;
        for (double value : v_values)
        {
            if (value == 0)
                continue;
            accum += pow(value - average, 2);
            total++;
        }
        return sqrt(accum / total);
    }

    void Tracking::LocalMapStats2File()
    {
        ofstream f;
        f.open("LocalMapTimeStats.txt");
        f << fixed << setprecision(6);
        f << "#Stereo rect[ms], MP culling[ms], MP creation[ms], LBA[ms], KF culling[ms], Total[ms]" << endl;
        for (int i = 0; i < mpLocalMapper->vdLMTotal_ms.size(); ++i)
        {
            f << mpLocalMapper->vdKFInsert_ms[i] << "," << mpLocalMapper->vdMPCulling_ms[i] << ","
              << mpLocalMapper->vdMPCreation_ms[i] << "," << mpLocalMapper->vdLBASync_ms[i] << ","
              << mpLocalMapper->vdKFCullingSync_ms[i] << "," << mpLocalMapper->vdLMTotal_ms[i] << endl;
        }

        f.close();

        f.open("LBA_Stats.txt");
        f << fixed << setprecision(6);
        f << "#LBA time[ms], KF opt[#], KF fixed[#], MP[#], Edges[#]" << endl;
        for (int i = 0; i < mpLocalMapper->vdLBASync_ms.size(); ++i)
        {
            f << mpLocalMapper->vdLBASync_ms[i] << "," << mpLocalMapper->vnLBA_KFopt[i] << ","
              << mpLocalMapper->vnLBA_KFfixed[i] << "," << mpLocalMapper->vnLBA_MPs[i] << ","
              << mpLocalMapper->vnLBA_edges[i] << endl;
        }

        f.close();
    }

    void Tracking::TrackStats2File()
    {
        ofstream f;
        f.open("SessionInfo.txt");
        f << fixed;
        f << "Number of KFs: " << mpAtlas->GetAllKeyFrames().size() << endl;
        f << "Number of MPs: " << mpAtlas->GetAllMapPoints().size() << endl;

        f << "OpenCV version: " << CV_VERSION << endl;

        f.close();

        f.open("TrackingTimeStats.txt");
        f << fixed << setprecision(6);

        f << "#Image Rect[ms], Image Resize[ms], ORB ext[ms], Stereo match[ms], IMU preint[ms], Pose pred[ms], LM track[ms], KF dec[ms], Total[ms]" << endl;

        for (int i = 0; i < vdTrackTotal_ms.size(); ++i)
        {
            double stereo_rect = 0.0;
            if (!vdRectStereo_ms.empty())
            {
                stereo_rect = vdRectStereo_ms[i];
            }

            double resize_image = 0.0;
            if (!vdResizeImage_ms.empty())
            {
                resize_image = vdResizeImage_ms[i];
            }

            double stereo_match = 0.0;
            if (!vdStereoMatch_ms.empty())
            {
                stereo_match = vdStereoMatch_ms[i];
            }

            double imu_preint = 0.0;
            if (!vdIMUInteg_ms.empty())
            {
                imu_preint = vdIMUInteg_ms[i];
            }

            f << stereo_rect << "," << resize_image << "," << vdORBExtract_ms[i] << "," << stereo_match << "," << imu_preint << ","
              << vdPosePred_ms[i] << "," << vdLMTrack_ms[i] << "," << vdNewKF_ms[i] << "," << vdTrackTotal_ms[i] << endl;
        }

        f.close();
    }

    void Tracking::PrintTimeStats()
    {
        // Save data in files
        TrackStats2File();
        LocalMapStats2File();

        ofstream f;
        f.open("ExecMean.txt");
        f << fixed;
        // Report the mean and std of each one
        std::cout << std::endl
                  << " TIME STATS in ms (mean$\\pm$std)" << std::endl;
        f << " TIME STATS in ms (mean$\\pm$std)" << std::endl;
        cout << "OpenCV version: " << CV_VERSION << endl;
        f << "OpenCV version: " << CV_VERSION << endl;
        std::cout << "---------------------------" << std::endl;
        std::cout << "Tracking" << std::setprecision(5) << std::endl
                  << std::endl;
        f << "---------------------------" << std::endl;
        f << "Tracking" << std::setprecision(5) << std::endl
          << std::endl;
        double average, deviation;
        if (!vdRectStereo_ms.empty())
        {
            average = calcAverage(vdRectStereo_ms);
            deviation = calcDeviation(vdRectStereo_ms, average);
            std::cout << "Stereo Rectification: " << average << "$\\pm$" << deviation << std::endl;
            f << "Stereo Rectification: " << average << "$\\pm$" << deviation << std::endl;
        }

        if (!vdResizeImage_ms.empty())
        {
            average = calcAverage(vdResizeImage_ms);
            deviation = calcDeviation(vdResizeImage_ms, average);
            std::cout << "Image Resize: " << average << "$\\pm$" << deviation << std::endl;
            f << "Image Resize: " << average << "$\\pm$" << deviation << std::endl;
        }

        average = calcAverage(vdORBExtract_ms);
        deviation = calcDeviation(vdORBExtract_ms, average);
        std::cout << "ORB Extraction: " << average << "$\\pm$" << deviation << std::endl;
        f << "ORB Extraction: " << average << "$\\pm$" << deviation << std::endl;

        if (!vdStereoMatch_ms.empty())
        {
            average = calcAverage(vdStereoMatch_ms);
            deviation = calcDeviation(vdStereoMatch_ms, average);
            std::cout << "Stereo Matching: " << average << "$\\pm$" << deviation << std::endl;
            f << "Stereo Matching: " << average << "$\\pm$" << deviation << std::endl;
        }

        if (!vdIMUInteg_ms.empty())
        {
            average = calcAverage(vdIMUInteg_ms);
            deviation = calcDeviation(vdIMUInteg_ms, average);
            std::cout << "IMU Preintegration: " << average << "$\\pm$" << deviation << std::endl;
            f << "IMU Preintegration: " << average << "$\\pm$" << deviation << std::endl;
        }

        average = calcAverage(vdPosePred_ms);
        deviation = calcDeviation(vdPosePred_ms, average);
        std::cout << "Pose Prediction: " << average << "$\\pm$" << deviation << std::endl;
        f << "Pose Prediction: " << average << "$\\pm$" << deviation << std::endl;

        average = calcAverage(vdLMTrack_ms);
        deviation = calcDeviation(vdLMTrack_ms, average);
        std::cout << "LM Track: " << average << "$\\pm$" << deviation << std::endl;
        f << "LM Track: " << average << "$\\pm$" << deviation << std::endl;

        average = calcAverage(vdNewKF_ms);
        deviation = calcDeviation(vdNewKF_ms, average);
        std::cout << "New KF decision: " << average << "$\\pm$" << deviation << std::endl;
        f << "New KF decision: " << average << "$\\pm$" << deviation << std::endl;

        average = calcAverage(vdTrackTotal_ms);
        deviation = calcDeviation(vdTrackTotal_ms, average);
        std::cout << "Total Tracking: " << average << "$\\pm$" << deviation << std::endl;
        f << "Total Tracking: " << average << "$\\pm$" << deviation << std::endl;

        // Local Mapping time stats
        std::cout << std::endl
                  << std::endl
                  << std::endl;
        std::cout << "Local Mapping" << std::endl
                  << std::endl;
        f << std::endl
          << "Local Mapping" << std::endl
          << std::endl;

        average = calcAverage(mpLocalMapper->vdKFInsert_ms);
        deviation = calcDeviation(mpLocalMapper->vdKFInsert_ms, average);
        std::cout << "KF Insertion: " << average << "$\\pm$" << deviation << std::endl;
        f << "KF Insertion: " << average << "$\\pm$" << deviation << std::endl;

        average = calcAverage(mpLocalMapper->vdMPCulling_ms);
        deviation = calcDeviation(mpLocalMapper->vdMPCulling_ms, average);
        std::cout << "MP Culling: " << average << "$\\pm$" << deviation << std::endl;
        f << "MP Culling: " << average << "$\\pm$" << deviation << std::endl;

        average = calcAverage(mpLocalMapper->vdMPCreation_ms);
        deviation = calcDeviation(mpLocalMapper->vdMPCreation_ms, average);
        std::cout << "MP Creation: " << average << "$\\pm$" << deviation << std::endl;
        f << "MP Creation: " << average << "$\\pm$" << deviation << std::endl;

        average = calcAverage(mpLocalMapper->vdLBA_ms);
        deviation = calcDeviation(mpLocalMapper->vdLBA_ms, average);
        std::cout << "LBA: " << average << "$\\pm$" << deviation << std::endl;
        f << "LBA: " << average << "$\\pm$" << deviation << std::endl;

        average = calcAverage(mpLocalMapper->vdKFCulling_ms);
        deviation = calcDeviation(mpLocalMapper->vdKFCulling_ms, average);
        std::cout << "KF Culling: " << average << "$\\pm$" << deviation << std::endl;
        f << "KF Culling: " << average << "$\\pm$" << deviation << std::endl;

        average = calcAverage(mpLocalMapper->vdLMTotal_ms);
        deviation = calcDeviation(mpLocalMapper->vdLMTotal_ms, average);
        std::cout << "Total Local Mapping: " << average << "$\\pm$" << deviation << std::endl;
        f << "Total Local Mapping: " << average << "$\\pm$" << deviation << std::endl;

        // Local Mapping LBA complexity
        std::cout << "---------------------------" << std::endl;
        std::cout << std::endl
                  << "LBA complexity (mean$\\pm$std)" << std::endl;
        f << "---------------------------" << std::endl;
        f << std::endl
          << "LBA complexity (mean$\\pm$std)" << std::endl;

        average = calcAverage(mpLocalMapper->vnLBA_edges);
        deviation = calcDeviation(mpLocalMapper->vnLBA_edges, average);
        std::cout << "LBA Edges: " << average << "$\\pm$" << deviation << std::endl;
        f << "LBA Edges: " << average << "$\\pm$" << deviation << std::endl;

        average = calcAverage(mpLocalMapper->vnLBA_KFopt);
        deviation = calcDeviation(mpLocalMapper->vnLBA_KFopt, average);
        std::cout << "LBA KF optimized: " << average << "$\\pm$" << deviation << std::endl;
        f << "LBA KF optimized: " << average << "$\\pm$" << deviation << std::endl;

        average = calcAverage(mpLocalMapper->vnLBA_KFfixed);
        deviation = calcDeviation(mpLocalMapper->vnLBA_KFfixed, average);
        std::cout << "LBA KF fixed: " << average << "$\\pm$" << deviation << std::endl;
        f << "LBA KF fixed: " << average << "$\\pm$" << deviation << std::endl;

        average = calcAverage(mpLocalMapper->vnLBA_MPs);
        deviation = calcDeviation(mpLocalMapper->vnLBA_MPs, average);
        std::cout << "LBA MP: " << average << "$\\pm$" << deviation << std::endl
                  << std::endl;
        f << "LBA MP: " << average << "$\\pm$" << deviation << std::endl
          << std::endl;

        std::cout << "LBA executions: " << mpLocalMapper->nLBA_exec << std::endl;
        std::cout << "LBA aborts: " << mpLocalMapper->nLBA_abort << std::endl;
        f << "LBA executions: " << mpLocalMapper->nLBA_exec << std::endl;
        f << "LBA aborts: " << mpLocalMapper->nLBA_abort << std::endl;

        // Map complexity
        std::cout << "---------------------------" << std::endl;
        std::cout << std::endl
                  << "Map complexity" << std::endl;
        std::cout << "KFs in map: " << mpAtlas->GetAllKeyFrames().size() << std::endl;
        std::cout << "MPs in map: " << mpAtlas->GetAllMapPoints().size() << std::endl;
        f << "---------------------------" << std::endl;
        f << std::endl
          << "Map complexity" << std::endl;
        vector<Map *> vpMaps = mpAtlas->GetAllMaps();
        Map *pBestMap = vpMaps[0];
        for (int i = 1; i < vpMaps.size(); ++i)
        {
            if (pBestMap->GetAllKeyFrames().size() < vpMaps[i]->GetAllKeyFrames().size())
            {
                pBestMap = vpMaps[i];
            }
        }

        f << "KFs in map: " << pBestMap->GetAllKeyFrames().size() << std::endl;
        f << "MPs in map: " << pBestMap->GetAllMapPoints().size() << std::endl;

        f << "---------------------------" << std::endl;
        f << std::endl
          << "Place Recognition (mean$\\pm$std)" << std::endl;
        std::cout << "---------------------------" << std::endl;
        std::cout << std::endl
                  << "Place Recognition (mean$\\pm$std)" << std::endl;
        average = calcAverage(mpLoopClosing->vdDataQuery_ms);
        deviation = calcDeviation(mpLoopClosing->vdDataQuery_ms, average);
        f << "Database Query: " << average << "$\\pm$" << deviation << std::endl;
        std::cout << "Database Query: " << average << "$\\pm$" << deviation << std::endl;
        average = calcAverage(mpLoopClosing->vdEstSim3_ms);
        deviation = calcDeviation(mpLoopClosing->vdEstSim3_ms, average);
        f << "SE3 estimation: " << average << "$\\pm$" << deviation << std::endl;
        std::cout << "SE3 estimation: " << average << "$\\pm$" << deviation << std::endl;
        average = calcAverage(mpLoopClosing->vdPRTotal_ms);
        deviation = calcDeviation(mpLoopClosing->vdPRTotal_ms, average);
        f << "Total Place Recognition: " << average << "$\\pm$" << deviation << std::endl
          << std::endl;
        std::cout << "Total Place Recognition: " << average << "$\\pm$" << deviation << std::endl
                  << std::endl;

        f << std::endl
          << "Loop Closing (mean$\\pm$std)" << std::endl;
        std::cout << std::endl
                  << "Loop Closing (mean$\\pm$std)" << std::endl;
        average = calcAverage(mpLoopClosing->vdLoopFusion_ms);
        deviation = calcDeviation(mpLoopClosing->vdLoopFusion_ms, average);
        f << "Loop Fusion: " << average << "$\\pm$" << deviation << std::endl;
        std::cout << "Loop Fusion: " << average << "$\\pm$" << deviation << std::endl;
        average = calcAverage(mpLoopClosing->vdLoopOptEss_ms);
        deviation = calcDeviation(mpLoopClosing->vdLoopOptEss_ms, average);
        f << "Essential Graph: " << average << "$\\pm$" << deviation << std::endl;
        std::cout << "Essential Graph: " << average << "$\\pm$" << deviation << std::endl;
        average = calcAverage(mpLoopClosing->vdLoopTotal_ms);
        deviation = calcDeviation(mpLoopClosing->vdLoopTotal_ms, average);
        f << "Total Loop Closing: " << average << "$\\pm$" << deviation << std::endl
          << std::endl;
        std::cout << "Total Loop Closing: " << average << "$\\pm$" << deviation << std::endl
                  << std::endl;

        f << "Numb exec: " << mpLoopClosing->nLoop << std::endl;
        std::cout << "Num exec: " << mpLoopClosing->nLoop << std::endl;
        average = calcAverage(mpLoopClosing->vnLoopKFs);
        deviation = calcDeviation(mpLoopClosing->vnLoopKFs, average);
        f << "Number of KFs: " << average << "$\\pm$" << deviation << std::endl;
        std::cout << "Number of KFs: " << average << "$\\pm$" << deviation << std::endl;

        f << std::endl
          << "Map Merging (mean$\\pm$std)" << std::endl;
        std::cout << std::endl
                  << "Map Merging (mean$\\pm$std)" << std::endl;
        average = calcAverage(mpLoopClosing->vdMergeMaps_ms);
        deviation = calcDeviation(mpLoopClosing->vdMergeMaps_ms, average);
        f << "Merge Maps: " << average << "$\\pm$" << deviation << std::endl;
        std::cout << "Merge Maps: " << average << "$\\pm$" << deviation << std::endl;
        average = calcAverage(mpLoopClosing->vdWeldingBA_ms);
        deviation = calcDeviation(mpLoopClosing->vdWeldingBA_ms, average);
        f << "Welding BA: " << average << "$\\pm$" << deviation << std::endl;
        std::cout << "Welding BA: " << average << "$\\pm$" << deviation << std::endl;
        average = calcAverage(mpLoopClosing->vdMergeOptEss_ms);
        deviation = calcDeviation(mpLoopClosing->vdMergeOptEss_ms, average);
        f << "Optimization Ess.: " << average << "$\\pm$" << deviation << std::endl;
        std::cout << "Optimization Ess.: " << average << "$\\pm$" << deviation << std::endl;
        average = calcAverage(mpLoopClosing->vdMergeTotal_ms);
        deviation = calcDeviation(mpLoopClosing->vdMergeTotal_ms, average);
        f << "Total Map Merging: " << average << "$\\pm$" << deviation << std::endl
          << std::endl;
        std::cout << "Total Map Merging: " << average << "$\\pm$" << deviation << std::endl
                  << std::endl;

        f << "Numb exec: " << mpLoopClosing->nMerges << std::endl;
        std::cout << "Num exec: " << mpLoopClosing->nMerges << std::endl;
        average = calcAverage(mpLoopClosing->vnMergeKFs);
        deviation = calcDeviation(mpLoopClosing->vnMergeKFs, average);
        f << "Number of KFs: " << average << "$\\pm$" << deviation << std::endl;
        std::cout << "Number of KFs: " << average << "$\\pm$" << deviation << std::endl;
        average = calcAverage(mpLoopClosing->vnMergeMPs);
        deviation = calcDeviation(mpLoopClosing->vnMergeMPs, average);
        f << "Number of MPs: " << average << "$\\pm$" << deviation << std::endl;
        std::cout << "Number of MPs: " << average << "$\\pm$" << deviation << std::endl;

        f << std::endl
          << "Full GBA (mean$\\pm$std)" << std::endl;
        std::cout << std::endl
                  << "Full GBA (mean$\\pm$std)" << std::endl;
        average = calcAverage(mpLoopClosing->vdGBA_ms);
        deviation = calcDeviation(mpLoopClosing->vdGBA_ms, average);
        f << "GBA: " << average << "$\\pm$" << deviation << std::endl;
        std::cout << "GBA: " << average << "$\\pm$" << deviation << std::endl;
        average = calcAverage(mpLoopClosing->vdUpdateMap_ms);
        deviation = calcDeviation(mpLoopClosing->vdUpdateMap_ms, average);
        f << "Map Update: " << average << "$\\pm$" << deviation << std::endl;
        std::cout << "Map Update: " << average << "$\\pm$" << deviation << std::endl;
        average = calcAverage(mpLoopClosing->vdFGBATotal_ms);
        deviation = calcDeviation(mpLoopClosing->vdFGBATotal_ms, average);
        f << "Total Full GBA: " << average << "$\\pm$" << deviation << std::endl
          << std::endl;
        std::cout << "Total Full GBA: " << average << "$\\pm$" << deviation << std::endl
                  << std::endl;

        f << "Numb exec: " << mpLoopClosing->nFGBA_exec << std::endl;
        std::cout << "Num exec: " << mpLoopClosing->nFGBA_exec << std::endl;
        f << "Numb abort: " << mpLoopClosing->nFGBA_abort << std::endl;
        std::cout << "Num abort: " << mpLoopClosing->nFGBA_abort << std::endl;
        average = calcAverage(mpLoopClosing->vnGBAKFs);
        deviation = calcDeviation(mpLoopClosing->vnGBAKFs, average);
        f << "Number of KFs: " << average << "$\\pm$" << deviation << std::endl;
        std::cout << "Number of KFs: " << average << "$\\pm$" << deviation << std::endl;
        average = calcAverage(mpLoopClosing->vnGBAMPs);
        deviation = calcDeviation(mpLoopClosing->vnGBAMPs, average);
        f << "Number of MPs: " << average << "$\\pm$" << deviation << std::endl;
        std::cout << "Number of MPs: " << average << "$\\pm$" << deviation << std::endl;

        f.close();
    }

#endif

    Tracking::~Tracking()
    {
        // f_track_stats.close();
    }

    void Tracking::newParameterLoader(Settings *settings)
    {
        mpCamera = settings->camera1();
        mpCamera = mpAtlas->AddCamera(mpCamera);

        if (settings->needToUndistort())
        {
            mDistCoef = settings->camera1DistortionCoef();
        }
        else
        {
            mDistCoef = cv::Mat::zeros(4, 1, CV_32F);
        }

        // TODO: missing image scaling and rectification
        mImageScale = 1.0f;

        mK = cv::Mat::eye(3, 3, CV_32F);
        mK.at<float>(0, 0) = mpCamera->getParameter(0);
        mK.at<float>(1, 1) = mpCamera->getParameter(1);
        mK.at<float>(0, 2) = mpCamera->getParameter(2);
        mK.at<float>(1, 2) = mpCamera->getParameter(3);

        mK_.setIdentity();
        mK_(0, 0) = mpCamera->getParameter(0);
        mK_(1, 1) = mpCamera->getParameter(1);
        mK_(0, 2) = mpCamera->getParameter(2);
        mK_(1, 2) = mpCamera->getParameter(3);

        if ((mSensor == System::STEREO || mSensor == System::IMU_STEREO || mSensor == System::IMU_RGBD) &&
            settings->cameraType() == Settings::KannalaBrandt)
        {
            mpCamera2 = settings->camera2();
            mpCamera2 = mpAtlas->AddCamera(mpCamera2);

            mTlr = settings->Tlr();

            mpFrameDrawer->both = true;
        }

        if (mSensor == System::STEREO || mSensor == System::RGBD || mSensor == System::IMU_STEREO || mSensor == System::IMU_RGBD)
        {
            mbf = settings->bf();
            mThDepth = settings->b() * settings->thDepth();
        }

        if (mSensor == System::RGBD || mSensor == System::IMU_RGBD)
        {
            mDepthMapFactor = settings->depthMapFactor();
            if (fabs(mDepthMapFactor) < 1e-5)
                mDepthMapFactor = 1;
            else
                mDepthMapFactor = 1.0f / mDepthMapFactor;
        }

        mMinFrames = settings->keyFrameMinFrames();
        mMaxFrames = settings->keyFrameMaxFrames();
        mbRGB = settings->rgb();

        // ORB parameters
        int nFeatures = settings->nFeatures();
        int nLevels = settings->nLevels();
        int fIniThFAST = settings->initThFAST();
        int fMinThFAST = settings->minThFAST();
        float fScaleFactor = settings->scaleFactor();

        mpORBextractorLeft = new ORBextractor(nFeatures, fScaleFactor, nLevels, fIniThFAST, fMinThFAST);

        if (mSensor == System::STEREO || mSensor == System::IMU_STEREO)
            mpORBextractorRight = new ORBextractor(nFeatures, fScaleFactor, nLevels, fIniThFAST, fMinThFAST);

        if (mSensor == System::MONOCULAR || mSensor == System::IMU_MONOCULAR)
            mpIniORBextractor = new ORBextractor(5 * nFeatures, fScaleFactor, nLevels, fIniThFAST, fMinThFAST);

        // IMU parameters
        Sophus::SE3f Tbc = settings->Tbc();
        mInsertKFsLost = settings->insertKFsWhenLost();
        mImuFreq = settings->imuFrequency();
        mImuPer = 0.001; // 1.0 / (double) mImuFreq;     //TODO: ESTO ESTA BIEN?
        float Ng = settings->noiseGyro();
        float Na = settings->noiseAcc();
        float Ngw = settings->gyroWalk();
        float Naw = settings->accWalk();

        const float sf = sqrt(mImuFreq);
        mpImuCalib = new IMU::Calib(Tbc, Ng * sf, Na * sf, Ngw / sf, Naw / sf);

        mpImuPreintegratedFromLastKF = new IMU::Preintegrated(IMU::Bias(), *mpImuCalib);
    }

    bool Tracking::ParseCamParamFile(cv::FileStorage &fSettings)
    {
        mDistCoef = cv::Mat::zeros(4, 1, CV_32F);
        cout << endl
             << "Camera Parameters: " << endl;
        bool b_miss_params = false;

        string sCameraName = fSettings["Camera.type"];
        if (sCameraName == "PinHole")
        {
            float fx, fy, cx, cy;
            mImageScale = 1.f;

            // Camera calibration parameters
            cv::FileNode node = fSettings["Camera.fx"];
            if (!node.empty() && node.isReal())
            {
                fx = node.real();
            }
            else
            {
                std::cerr << "*Camera.fx parameter doesn't exist or is not a real number*" << std::endl;
                b_miss_params = true;
            }

            node = fSettings["Camera.fy"];
            if (!node.empty() && node.isReal())
            {
                fy = node.real();
            }
            else
            {
                std::cerr << "*Camera.fy parameter doesn't exist or is not a real number*" << std::endl;
                b_miss_params = true;
            }

            node = fSettings["Camera.cx"];
            if (!node.empty() && node.isReal())
            {
                cx = node.real();
            }
            else
            {
                std::cerr << "*Camera.cx parameter doesn't exist or is not a real number*" << std::endl;
                b_miss_params = true;
            }

            node = fSettings["Camera.cy"];
            if (!node.empty() && node.isReal())
            {
                cy = node.real();
            }
            else
            {
                std::cerr << "*Camera.cy parameter doesn't exist or is not a real number*" << std::endl;
                b_miss_params = true;
            }

            // Distortion parameters
            node = fSettings["Camera.k1"];
            if (!node.empty() && node.isReal())
            {
                mDistCoef.at<float>(0) = node.real();
            }
            else
            {
                std::cerr << "*Camera.k1 parameter doesn't exist or is not a real number*" << std::endl;
                b_miss_params = true;
            }

            node = fSettings["Camera.k2"];
            if (!node.empty() && node.isReal())
            {
                mDistCoef.at<float>(1) = node.real();
            }
            else
            {
                std::cerr << "*Camera.k2 parameter doesn't exist or is not a real number*" << std::endl;
                b_miss_params = true;
            }

            node = fSettings["Camera.p1"];
            if (!node.empty() && node.isReal())
            {
                mDistCoef.at<float>(2) = node.real();
            }
            else
            {
                std::cerr << "*Camera.p1 parameter doesn't exist or is not a real number*" << std::endl;
                b_miss_params = true;
            }

            node = fSettings["Camera.p2"];
            if (!node.empty() && node.isReal())
            {
                mDistCoef.at<float>(3) = node.real();
            }
            else
            {
                std::cerr << "*Camera.p2 parameter doesn't exist or is not a real number*" << std::endl;
                b_miss_params = true;
            }

            node = fSettings["Camera.k3"];
            if (!node.empty() && node.isReal())
            {
                mDistCoef.resize(5);
                mDistCoef.at<float>(4) = node.real();
            }

            node = fSettings["Camera.imageScale"];
            if (!node.empty() && node.isReal())
            {
                mImageScale = node.real();
            }

            if (b_miss_params)
            {
                return false;
            }

            if (mImageScale != 1.f)
            {
                // K matrix parameters must be scaled.
                fx = fx * mImageScale;
                fy = fy * mImageScale;
                cx = cx * mImageScale;
                cy = cy * mImageScale;
            }

            vector<float> vCamCalib{fx, fy, cx, cy};

            mpCamera = new Pinhole(vCamCalib);

            mpCamera = mpAtlas->AddCamera(mpCamera);

            std::cout << "- Camera: Pinhole" << std::endl;
            std::cout << "- Image scale: " << mImageScale << std::endl;
            std::cout << "- fx: " << fx << std::endl;
            std::cout << "- fy: " << fy << std::endl;
            std::cout << "- cx: " << cx << std::endl;
            std::cout << "- cy: " << cy << std::endl;
            std::cout << "- k1: " << mDistCoef.at<float>(0) << std::endl;
            std::cout << "- k2: " << mDistCoef.at<float>(1) << std::endl;

            std::cout << "- p1: " << mDistCoef.at<float>(2) << std::endl;
            std::cout << "- p2: " << mDistCoef.at<float>(3) << std::endl;

            if (mDistCoef.rows == 5)
                std::cout << "- k3: " << mDistCoef.at<float>(4) << std::endl;

            mK = cv::Mat::eye(3, 3, CV_32F);
            mK.at<float>(0, 0) = fx;
            mK.at<float>(1, 1) = fy;
            mK.at<float>(0, 2) = cx;
            mK.at<float>(1, 2) = cy;

            mK_.setIdentity();
            mK_(0, 0) = fx;
            mK_(1, 1) = fy;
            mK_(0, 2) = cx;
            mK_(1, 2) = cy;
        }
        else if (sCameraName == "KannalaBrandt8")
        {
            float fx, fy, cx, cy;
            float k1, k2, k3, k4;
            mImageScale = 1.f;

            // Camera calibration parameters
            cv::FileNode node = fSettings["Camera.fx"];
            if (!node.empty() && node.isReal())
            {
                fx = node.real();
            }
            else
            {
                std::cerr << "*Camera.fx parameter doesn't exist or is not a real number*" << std::endl;
                b_miss_params = true;
            }
            node = fSettings["Camera.fy"];
            if (!node.empty() && node.isReal())
            {
                fy = node.real();
            }
            else
            {
                std::cerr << "*Camera.fy parameter doesn't exist or is not a real number*" << std::endl;
                b_miss_params = true;
            }

            node = fSettings["Camera.cx"];
            if (!node.empty() && node.isReal())
            {
                cx = node.real();
            }
            else
            {
                std::cerr << "*Camera.cx parameter doesn't exist or is not a real number*" << std::endl;
                b_miss_params = true;
            }

            node = fSettings["Camera.cy"];
            if (!node.empty() && node.isReal())
            {
                cy = node.real();
            }
            else
            {
                std::cerr << "*Camera.cy parameter doesn't exist or is not a real number*" << std::endl;
                b_miss_params = true;
            }

            // Distortion parameters
            node = fSettings["Camera.k1"];
            if (!node.empty() && node.isReal())
            {
                k1 = node.real();
            }
            else
            {
                std::cerr << "*Camera.k1 parameter doesn't exist or is not a real number*" << std::endl;
                b_miss_params = true;
            }
            node = fSettings["Camera.k2"];
            if (!node.empty() && node.isReal())
            {
                k2 = node.real();
            }
            else
            {
                std::cerr << "*Camera.k2 parameter doesn't exist or is not a real number*" << std::endl;
                b_miss_params = true;
            }

            node = fSettings["Camera.k3"];
            if (!node.empty() && node.isReal())
            {
                k3 = node.real();
            }
            else
            {
                std::cerr << "*Camera.k3 parameter doesn't exist or is not a real number*" << std::endl;
                b_miss_params = true;
            }

            node = fSettings["Camera.k4"];
            if (!node.empty() && node.isReal())
            {
                k4 = node.real();
            }
            else
            {
                std::cerr << "*Camera.k4 parameter doesn't exist or is not a real number*" << std::endl;
                b_miss_params = true;
            }

            node = fSettings["Camera.imageScale"];
            if (!node.empty() && node.isReal())
            {
                mImageScale = node.real();
            }

            if (!b_miss_params)
            {
                if (mImageScale != 1.f)
                {
                    // K matrix parameters must be scaled.
                    fx = fx * mImageScale;
                    fy = fy * mImageScale;
                    cx = cx * mImageScale;
                    cy = cy * mImageScale;
                }

                vector<float> vCamCalib{fx, fy, cx, cy, k1, k2, k3, k4};
                mpCamera = new KannalaBrandt8(vCamCalib);
                mpCamera = mpAtlas->AddCamera(mpCamera);
                std::cout << "- Camera: Fisheye" << std::endl;
                std::cout << "- Image scale: " << mImageScale << std::endl;
                std::cout << "- fx: " << fx << std::endl;
                std::cout << "- fy: " << fy << std::endl;
                std::cout << "- cx: " << cx << std::endl;
                std::cout << "- cy: " << cy << std::endl;
                std::cout << "- k1: " << k1 << std::endl;
                std::cout << "- k2: " << k2 << std::endl;
                std::cout << "- k3: " << k3 << std::endl;
                std::cout << "- k4: " << k4 << std::endl;

                mK = cv::Mat::eye(3, 3, CV_32F);
                mK.at<float>(0, 0) = fx;
                mK.at<float>(1, 1) = fy;
                mK.at<float>(0, 2) = cx;
                mK.at<float>(1, 2) = cy;

                mK_.setIdentity();
                mK_(0, 0) = fx;
                mK_(1, 1) = fy;
                mK_(0, 2) = cx;
                mK_(1, 2) = cy;
            }

            if (mSensor == System::STEREO || mSensor == System::IMU_STEREO || mSensor == System::IMU_RGBD)
            {
                // Right camera
                // Camera calibration parameters
                cv::FileNode node = fSettings["Camera2.fx"];
                if (!node.empty() && node.isReal())
                {
                    fx = node.real();
                }
                else
                {
                    std::cerr << "*Camera2.fx parameter doesn't exist or is not a real number*" << std::endl;
                    b_miss_params = true;
                }
                node = fSettings["Camera2.fy"];
                if (!node.empty() && node.isReal())
                {
                    fy = node.real();
                }
                else
                {
                    std::cerr << "*Camera2.fy parameter doesn't exist or is not a real number*" << std::endl;
                    b_miss_params = true;
                }

                node = fSettings["Camera2.cx"];
                if (!node.empty() && node.isReal())
                {
                    cx = node.real();
                }
                else
                {
                    std::cerr << "*Camera2.cx parameter doesn't exist or is not a real number*" << std::endl;
                    b_miss_params = true;
                }

                node = fSettings["Camera2.cy"];
                if (!node.empty() && node.isReal())
                {
                    cy = node.real();
                }
                else
                {
                    std::cerr << "*Camera2.cy parameter doesn't exist or is not a real number*" << std::endl;
                    b_miss_params = true;
                }

                // Distortion parameters
                node = fSettings["Camera2.k1"];
                if (!node.empty() && node.isReal())
                {
                    k1 = node.real();
                }
                else
                {
                    std::cerr << "*Camera2.k1 parameter doesn't exist or is not a real number*" << std::endl;
                    b_miss_params = true;
                }
                node = fSettings["Camera2.k2"];
                if (!node.empty() && node.isReal())
                {
                    k2 = node.real();
                }
                else
                {
                    std::cerr << "*Camera2.k2 parameter doesn't exist or is not a real number*" << std::endl;
                    b_miss_params = true;
                }

                node = fSettings["Camera2.k3"];
                if (!node.empty() && node.isReal())
                {
                    k3 = node.real();
                }
                else
                {
                    std::cerr << "*Camera2.k3 parameter doesn't exist or is not a real number*" << std::endl;
                    b_miss_params = true;
                }

                node = fSettings["Camera2.k4"];
                if (!node.empty() && node.isReal())
                {
                    k4 = node.real();
                }
                else
                {
                    std::cerr << "*Camera2.k4 parameter doesn't exist or is not a real number*" << std::endl;
                    b_miss_params = true;
                }

                int leftLappingBegin = -1;
                int leftLappingEnd = -1;

                int rightLappingBegin = -1;
                int rightLappingEnd = -1;

                node = fSettings["Camera.lappingBegin"];
                if (!node.empty() && node.isInt())
                {
                    leftLappingBegin = node.operator int();
                }
                else
                {
                    std::cout << "WARNING: Camera.lappingBegin not correctly defined" << std::endl;
                }
                node = fSettings["Camera.lappingEnd"];
                if (!node.empty() && node.isInt())
                {
                    leftLappingEnd = node.operator int();
                }
                else
                {
                    std::cout << "WARNING: Camera.lappingEnd not correctly defined" << std::endl;
                }
                node = fSettings["Camera2.lappingBegin"];
                if (!node.empty() && node.isInt())
                {
                    rightLappingBegin = node.operator int();
                }
                else
                {
                    std::cout << "WARNING: Camera2.lappingBegin not correctly defined" << std::endl;
                }
                node = fSettings["Camera2.lappingEnd"];
                if (!node.empty() && node.isInt())
                {
                    rightLappingEnd = node.operator int();
                }
                else
                {
                    std::cout << "WARNING: Camera2.lappingEnd not correctly defined" << std::endl;
                }

                node = fSettings["Tlr"];
                cv::Mat cvTlr;
                if (!node.empty())
                {
                    cvTlr = node.mat();
                    if (cvTlr.rows != 3 || cvTlr.cols != 4)
                    {
                        std::cerr << "*Tlr matrix have to be a 3x4 transformation matrix*" << std::endl;
                        b_miss_params = true;
                    }
                }
                else
                {
                    std::cerr << "*Tlr matrix doesn't exist*" << std::endl;
                    b_miss_params = true;
                }

                if (!b_miss_params)
                {
                    if (mImageScale != 1.f)
                    {
                        // K matrix parameters must be scaled.
                        fx = fx * mImageScale;
                        fy = fy * mImageScale;
                        cx = cx * mImageScale;
                        cy = cy * mImageScale;

                        leftLappingBegin = leftLappingBegin * mImageScale;
                        leftLappingEnd = leftLappingEnd * mImageScale;
                        rightLappingBegin = rightLappingBegin * mImageScale;
                        rightLappingEnd = rightLappingEnd * mImageScale;
                    }

                    static_cast<KannalaBrandt8 *>(mpCamera)->mvLappingArea[0] = leftLappingBegin;
                    static_cast<KannalaBrandt8 *>(mpCamera)->mvLappingArea[1] = leftLappingEnd;

                    mpFrameDrawer->both = true;

                    vector<float> vCamCalib2{fx, fy, cx, cy, k1, k2, k3, k4};
                    mpCamera2 = new KannalaBrandt8(vCamCalib2);
                    mpCamera2 = mpAtlas->AddCamera(mpCamera2);

                    mTlr = Converter::toSophus(cvTlr);

                    static_cast<KannalaBrandt8 *>(mpCamera2)->mvLappingArea[0] = rightLappingBegin;
                    static_cast<KannalaBrandt8 *>(mpCamera2)->mvLappingArea[1] = rightLappingEnd;

                    std::cout << "- Camera1 Lapping: " << leftLappingBegin << ", " << leftLappingEnd << std::endl;

                    std::cout << std::endl
                              << "Camera2 Parameters:" << std::endl;
                    std::cout << "- Camera: Fisheye" << std::endl;
                    std::cout << "- Image scale: " << mImageScale << std::endl;
                    std::cout << "- fx: " << fx << std::endl;
                    std::cout << "- fy: " << fy << std::endl;
                    std::cout << "- cx: " << cx << std::endl;
                    std::cout << "- cy: " << cy << std::endl;
                    std::cout << "- k1: " << k1 << std::endl;
                    std::cout << "- k2: " << k2 << std::endl;
                    std::cout << "- k3: " << k3 << std::endl;
                    std::cout << "- k4: " << k4 << std::endl;

                    std::cout << "- mTlr: \n"
                              << cvTlr << std::endl;

                    std::cout << "- Camera2 Lapping: " << rightLappingBegin << ", " << rightLappingEnd << std::endl;
                }
            }

            if (b_miss_params)
            {
                return false;
            }
        }
        else
        {
            std::cerr << "*Not Supported Camera Sensor*" << std::endl;
            std::cerr << "Check an example configuration file with the desired sensor" << std::endl;
        }

        if (mSensor == System::STEREO || mSensor == System::RGBD || mSensor == System::IMU_STEREO || mSensor == System::IMU_RGBD)
        {
            cv::FileNode node = fSettings["Camera.bf"];
            if (!node.empty() && node.isReal())
            {
                mbf = node.real();
                if (mImageScale != 1.f)
                {
                    mbf *= mImageScale;
                }
            }
            else
            {
                std::cerr << "*Camera.bf parameter doesn't exist or is not a real number*" << std::endl;
                b_miss_params = true;
            }
        }

        float fps = fSettings["Camera.fps"];
        if (fps == 0)
            fps = 30;

        // Max/Min Frames to insert keyframes and to check relocalisation
        mMinFrames = 0;
        mMaxFrames = fps;

        cout << "- fps: " << fps << endl;

        cv::FileNode node_kf = fSettings["System.KeyFrameMinFrames"];
        if (!node_kf.empty() && node_kf.isInt())
            mMinFrames = (int)node_kf;
        node_kf = fSettings["System.KeyFrameMaxFrames"];
        if (!node_kf.empty() && node_kf.isInt())
            mMaxFrames = (int)node_kf;

        int nRGB = fSettings["Camera.RGB"];
        mbRGB = nRGB;

        if (mbRGB)
            cout << "- color order: RGB (ignored if grayscale)" << endl;
        else
            cout << "- color order: BGR (ignored if grayscale)" << endl;

        if (mSensor == System::STEREO || mSensor == System::RGBD || mSensor == System::IMU_STEREO || mSensor == System::IMU_RGBD)
        {
            float fx = mpCamera->getParameter(0);
            cv::FileNode node = fSettings["ThDepth"];
            if (!node.empty() && node.isReal())
            {
                mThDepth = node.real();
                mThDepth = mbf * mThDepth / fx;
                cout << endl
                     << "Depth Threshold (Close/Far Points): " << mThDepth << endl;
            }
            else
            {
                std::cerr << "*ThDepth parameter doesn't exist or is not a real number*" << std::endl;
                b_miss_params = true;
            }
        }

        if (mSensor == System::RGBD || mSensor == System::IMU_RGBD)
        {
            cv::FileNode node = fSettings["DepthMapFactor"];
            if (!node.empty() && node.isReal())
            {
                mDepthMapFactor = node.real();
                if (fabs(mDepthMapFactor) < 1e-5)
                    mDepthMapFactor = 1;
                else
                    mDepthMapFactor = 1.0f / mDepthMapFactor;
            }
            else
            {
                std::cerr << "*DepthMapFactor parameter doesn't exist or is not a real number*" << std::endl;
                b_miss_params = true;
            }
        }

        if (b_miss_params)
        {
            return false;
        }

        return true;
    }

    bool Tracking::ParseORBParamFile(cv::FileStorage &fSettings)
    {
        bool b_miss_params = false;
        int nFeatures, nLevels, fIniThFAST, fMinThFAST;
        float fScaleFactor;

        cv::FileNode node = fSettings["ORBextractor.nFeatures"];
        if (!node.empty() && node.isInt())
        {
            nFeatures = node.operator int();
        }
        else
        {
            std::cerr << "*ORBextractor.nFeatures parameter doesn't exist or is not an integer*" << std::endl;
            b_miss_params = true;
        }

        node = fSettings["ORBextractor.scaleFactor"];
        if (!node.empty() && node.isReal())
        {
            fScaleFactor = node.real();
        }
        else
        {
            std::cerr << "*ORBextractor.scaleFactor parameter doesn't exist or is not a real number*" << std::endl;
            b_miss_params = true;
        }

        node = fSettings["ORBextractor.nLevels"];
        if (!node.empty() && node.isInt())
        {
            nLevels = node.operator int();
        }
        else
        {
            std::cerr << "*ORBextractor.nLevels parameter doesn't exist or is not an integer*" << std::endl;
            b_miss_params = true;
        }

        node = fSettings["ORBextractor.iniThFAST"];
        if (!node.empty() && node.isInt())
        {
            fIniThFAST = node.operator int();
        }
        else
        {
            std::cerr << "*ORBextractor.iniThFAST parameter doesn't exist or is not an integer*" << std::endl;
            b_miss_params = true;
        }

        node = fSettings["ORBextractor.minThFAST"];
        if (!node.empty() && node.isInt())
        {
            fMinThFAST = node.operator int();
        }
        else
        {
            std::cerr << "*ORBextractor.minThFAST parameter doesn't exist or is not an integer*" << std::endl;
            b_miss_params = true;
        }

        if (b_miss_params)
        {
            return false;
        }

        mpORBextractorLeft = new ORBextractor(nFeatures, fScaleFactor, nLevels, fIniThFAST, fMinThFAST);

        if (mSensor == System::STEREO || mSensor == System::IMU_STEREO)
            mpORBextractorRight = new ORBextractor(nFeatures, fScaleFactor, nLevels, fIniThFAST, fMinThFAST);

        if (mSensor == System::MONOCULAR || mSensor == System::IMU_MONOCULAR)
            mpIniORBextractor = new ORBextractor(5 * nFeatures, fScaleFactor, nLevels, fIniThFAST, fMinThFAST);

        cout << endl
             << "ORB Extractor Parameters: " << endl;
        cout << "- Number of Features: " << nFeatures << endl;
        cout << "- Scale Levels: " << nLevels << endl;
        cout << "- Scale Factor: " << fScaleFactor << endl;
        cout << "- Initial Fast Threshold: " << fIniThFAST << endl;
        cout << "- Minimum Fast Threshold: " << fMinThFAST << endl;

        return true;
    }

    bool Tracking::ParseIMUParamFile(cv::FileStorage &fSettings)
    {
        bool b_miss_params = false;

        cv::Mat cvTbc;
        cv::FileNode node = fSettings["Tbc"];
        if (!node.empty())
        {
            cvTbc = node.mat();
            if (cvTbc.rows != 4 || cvTbc.cols != 4)
            {
                std::cerr << "*Tbc matrix have to be a 4x4 transformation matrix*" << std::endl;
                b_miss_params = true;
            }
        }
        else
        {
            std::cerr << "*Tbc matrix doesn't exist*" << std::endl;
            b_miss_params = true;
        }
        cout << endl;
        cout << "Left camera to Imu Transform (Tbc): " << endl
             << cvTbc << endl;
        Eigen::Matrix<float, 4, 4, Eigen::RowMajor> eigTbc(cvTbc.ptr<float>(0));
        Sophus::SE3f Tbc(eigTbc);

        node = fSettings["InsertKFsWhenLost"];
        mInsertKFsLost = true;
        if (!node.empty() && node.isInt())
        {
            mInsertKFsLost = (bool)node.operator int();
        }

        if (!mInsertKFsLost)
            cout << "Do not insert keyframes when lost visual tracking " << endl;

        float Ng, Na, Ngw, Naw;

        node = fSettings["IMU.Frequency"];
        if (!node.empty() && node.isInt())
        {
            mImuFreq = node.operator int();
            mImuPer = 0.001; // 1.0 / (double) mImuFreq;
        }
        else
        {
            std::cerr << "*IMU.Frequency parameter doesn't exist or is not an integer*" << std::endl;
            b_miss_params = true;
        }

        node = fSettings["IMU.NoiseGyro"];
        if (!node.empty() && node.isReal())
        {
            Ng = node.real();
        }
        else
        {
            std::cerr << "*IMU.NoiseGyro parameter doesn't exist or is not a real number*" << std::endl;
            b_miss_params = true;
        }

        node = fSettings["IMU.NoiseAcc"];
        if (!node.empty() && node.isReal())
        {
            Na = node.real();
        }
        else
        {
            std::cerr << "*IMU.NoiseAcc parameter doesn't exist or is not a real number*" << std::endl;
            b_miss_params = true;
        }

        node = fSettings["IMU.GyroWalk"];
        if (!node.empty() && node.isReal())
        {
            Ngw = node.real();
        }
        else
        {
            std::cerr << "*IMU.GyroWalk parameter doesn't exist or is not a real number*" << std::endl;
            b_miss_params = true;
        }

        node = fSettings["IMU.AccWalk"];
        if (!node.empty() && node.isReal())
        {
            Naw = node.real();
        }
        else
        {
            std::cerr << "*IMU.AccWalk parameter doesn't exist or is not a real number*" << std::endl;
            b_miss_params = true;
        }

        node = fSettings["IMU.fastInit"];
        mFastInit = false;
        if (!node.empty())
        {
            mFastInit = static_cast<int>(fSettings["IMU.fastInit"]) != 0;
        }

        if (mFastInit)
            cout << "Fast IMU initialization. Acceleration is not checked \n";

        if (b_miss_params)
        {
            return false;
        }

        const float sf = sqrt(mImuFreq);
        cout << endl;
        cout << "IMU frequency: " << mImuFreq << " Hz" << endl;
        cout << "IMU gyro noise: " << Ng << " rad/s/sqrt(Hz)" << endl;
        cout << "IMU gyro walk: " << Ngw << " rad/s^2/sqrt(Hz)" << endl;
        cout << "IMU accelerometer noise: " << Na << " m/s^2/sqrt(Hz)" << endl;
        cout << "IMU accelerometer walk: " << Naw << " m/s^3/sqrt(Hz)" << endl;

        mpImuCalib = new IMU::Calib(Tbc, Ng * sf, Na * sf, Ngw / sf, Naw / sf);

        mpImuPreintegratedFromLastKF = new IMU::Preintegrated(IMU::Bias(), *mpImuCalib);

        return true;
    }

    void Tracking::SetLocalMapper(LocalMapping *pLocalMapper)
    {
        mpLocalMapper = pLocalMapper;
    }

    void Tracking::SetLoopClosing(LoopClosing *pLoopClosing)
    {
        mpLoopClosing = pLoopClosing;
    }

    void Tracking::SetViewer(Viewer *pViewer)
    {
        mpViewer = pViewer;
    }

    void Tracking::SetDetector(YoloDetection *pDetector)
    {
        mpDetector = pDetector;
    }
    
    void Tracking::SetPointCloudMapping(PointCloudMapping *pPointCloudMapping)
    {
        mpPointCloudMapping = pPointCloudMapping;
    }

    void Tracking::SetStepByStep(bool bSet)
    {
        bStepByStep = bSet;
    }

    bool Tracking::GetStepByStep()
    {
        return bStepByStep;
    }

    Sophus::SE3f Tracking::GrabImageStereo(const cv::Mat &imRectLeft, const cv::Mat &imRectRight, const double &timestamp, string filename)
    {
        // cout << "GrabImageStereo" << endl;

        // Store previous frame image for Optical Flow
        if(!mImGray.empty())
            mLastImGray = mImGray.clone();

        mImGray = imRectLeft;
        cv::Mat imGrayRight = imRectRight;
        mImRight = imRectRight;

        if (mImGray.channels() == 3)
        {
            // cout << "Image with 3 channels" << endl;
            if (mbRGB)
            {
                cvtColor(mImGray, mImGray, cv::COLOR_RGB2GRAY);
                cvtColor(imGrayRight, imGrayRight, cv::COLOR_RGB2GRAY);
            }
            else
            {
                cvtColor(mImGray, mImGray, cv::COLOR_BGR2GRAY);
                cvtColor(imGrayRight, imGrayRight, cv::COLOR_BGR2GRAY);
            }
        }
        else if (mImGray.channels() == 4)
        {
            // cout << "Image with 4 channels" << endl;
            if (mbRGB)
            {
                cvtColor(mImGray, mImGray, cv::COLOR_RGBA2GRAY);
                cvtColor(imGrayRight, imGrayRight, cv::COLOR_RGBA2GRAY);
            }
            else
            {
                cvtColor(mImGray, mImGray, cv::COLOR_BGRA2GRAY);
                cvtColor(imGrayRight, imGrayRight, cv::COLOR_BGRA2GRAY);
            }
        }

        // cout << "Incoming frame creation" << endl;

        if (mSensor == System::STEREO && !mpCamera2)
            mCurrentFrame = Frame(mImGray, imGrayRight, timestamp, mpORBextractorLeft, mpORBextractorRight, mpORBVocabulary, mK, mDistCoef, mbf, mThDepth, mpCamera);
        else if (mSensor == System::STEREO && mpCamera2)
            mCurrentFrame = Frame(mImGray, imGrayRight, timestamp, mpORBextractorLeft, mpORBextractorRight, mpORBVocabulary, mK, mDistCoef, mbf, mThDepth, mpCamera, mpCamera2, mTlr);
        else if (mSensor == System::IMU_STEREO && !mpCamera2)
            mCurrentFrame = Frame(mImGray, imGrayRight, timestamp, mpORBextractorLeft, mpORBextractorRight, mpORBVocabulary, mK, mDistCoef, mbf, mThDepth, mpCamera, &mLastFrame, *mpImuCalib);
        else if (mSensor == System::IMU_STEREO && mpCamera2)
            mCurrentFrame = Frame(mImGray, imGrayRight, timestamp, mpORBextractorLeft, mpORBextractorRight, mpORBVocabulary, mK, mDistCoef, mbf, mThDepth, mpCamera, mpCamera2, mTlr, &mLastFrame, *mpImuCalib);

        // RDFP-SLAM: Check Optical Flow Consistency immediately after Frame creation
        if(!mLastImGray.empty())
            CheckFlowConsistency();

        // cout << "Incoming frame ended" << endl;

        mCurrentFrame.mNameFile = filename;
        mCurrentFrame.mnDataset = mnNumDataset;

#ifdef REGISTER_TIMES
        vdORBExtract_ms.push_back(mCurrentFrame.mTimeORB_Ext);
        vdStereoMatch_ms.push_back(mCurrentFrame.mTimeStereoMatch);
#endif

        // cout << "Tracking start" << endl;
        Track();
        // cout << "Tracking end" << endl;

        return mCurrentFrame.GetPose();
    }

    Sophus::SE3f Tracking::GrabImageRGBD(const cv::Mat &imRGB,const cv::Mat &imD, const double &timestamp, string filename)
    {
            // End-to-end pipeline profiling (per-frame stage timing)
            const bool ydm_pipe_profile = (std::getenv("YDM_PIPELINE_PROFILE") && std::string(std::getenv("YDM_PIPELINE_PROFILE")) == "1");
            int ydm_pipe_every = 1;
            if (const char* env_pe = std::getenv("YDM_PIPELINE_PROFILE_EVERY"))
            {
                try { ydm_pipe_every = std::stoi(std::string(env_pe)); } catch (...) { ydm_pipe_every = 1; }
                if (ydm_pipe_every < 1) ydm_pipe_every = 1;
            }
            std::chrono::steady_clock::time_point ydm_pipe_t0;
            std::chrono::steady_clock::time_point ydm_pipe_t;
            auto ydm_pipe_ms_since = [](const std::chrono::steady_clock::time_point &a,
                                        const std::chrono::steady_clock::time_point &b) -> double {
                return std::chrono::duration_cast<std::chrono::duration<double, std::milli>>(b - a).count();
            };
            double ydm_ms_prev = 0.0, ydm_ms_yolo = 0.0, ydm_ms_gray = 0.0, ydm_ms_depth = 0.0;
            double ydm_ms_frame = 0.0, ydm_ms_track = 0.0, ydm_ms_cleanup = 0.0;
            if (ydm_pipe_profile) { ydm_pipe_t0 = ydm_pipe_t = std::chrono::steady_clock::now(); }

        // Store previous frame image for Optical Flow
        if(!mImGray.empty())
            mLastImGray = mImGray.clone();

        // Shift previous depth/mask state for DIO-style processing
        if(!mCurrDepthMeters.empty())
            mPrevDepthMeters = mCurrDepthMeters.clone();
        if(!mCurrDynamicRigidMask.empty())
            mPrevDynamicRigidMask = mCurrDynamicRigidMask.clone();
        mCurrDynamicRigidMask.release();

        mImGray = imRGB;
        cv::Mat imDepth = imD;
        
        // 保存RGB和深度图用于点云生成
        if(mSensor == System::RGBD || mSensor == System::IMU_RGBD)
        {
            mImRGB = imRGB.clone();
            mImDepth = imD.clone();
        }

        if (ydm_pipe_profile)
        {
            const auto t1 = std::chrono::steady_clock::now();
            ydm_ms_prev = ydm_pipe_ms_since(ydm_pipe_t, t1);
            ydm_pipe_t = t1;
        }

        // Yolo
        cv::Mat InputImage;
        InputImage = imRGB.clone();
        mpDetector->GetImage(InputImage);
        // Always run YOLO every frame.
        mpDetector->Detect();

        if (ydm_pipe_profile)
        {
            const auto t1 = std::chrono::steady_clock::now();
            ydm_ms_yolo = ydm_pipe_ms_since(ydm_pipe_t, t1);
            ydm_pipe_t = t1;
        }

        // Cache instance-level outputs for DIO-SLAM style logic
        mCurrInstanceIdMask = mpDetector->mInstanceIdMask.empty() ? cv::Mat() : mpDetector->mInstanceIdMask.clone();
        mCurrInstanceClassNames = mpDetector->mvInstanceClassNames;

        // Persist results beyond this function call (mpDetector outputs are cleared later).
        mpORBextractorLeft->mvDynamicMask = mpDetector->mvDynamicMask;
        mpORBextractorLeft->mvDynamicArea = mpDetector->mvDynamicArea;
        if (mpViewer)
        {
            std::unique_lock<std::mutex> lock(mpViewer->mMutexPAFinsh);
            mpViewer->mmDetectMap = mpDetector->mmDetectMap;
            mpViewer->mask = mpDetector->mask;
        }

        // 灰度转换
        if (mImGray.channels() == 3)
        {
            if (mbRGB)
                cvtColor(mImGray, mImGray, cv::COLOR_RGB2GRAY);
            else
                cvtColor(mImGray, mImGray, cv::COLOR_BGR2GRAY);
        }
        else if (mImGray.channels() == 4)
        {
            if (mbRGB)
                cvtColor(mImGray, mImGray, cv::COLOR_RGBA2GRAY);
            else
                cvtColor(mImGray, mImGray, cv::COLOR_BGRA2GRAY);
        }

        if (ydm_pipe_profile)
        {
            const auto t1 = std::chrono::steady_clock::now();
            ydm_ms_gray = ydm_pipe_ms_since(ydm_pipe_t, t1);
            ydm_pipe_t = t1;
        }

        if ((fabs(mDepthMapFactor - 1.0f) > 1e-5) || imDepth.type() != CV_32F)
            imDepth.convertTo(imDepth, CV_32F, mDepthMapFactor);

        // Cache current depth (meters) for DIO-SLAM style ego-motion compensation
        mCurrDepthMeters = imDepth.clone();

        if (ydm_pipe_profile)
        {
            const auto t1 = std::chrono::steady_clock::now();
            ydm_ms_depth = ydm_pipe_ms_since(ydm_pipe_t, t1);
            ydm_pipe_t = t1;
        }

        if (mSensor == System::RGBD)
            mCurrentFrame = Frame(mImGray, imDepth, timestamp, mpORBextractorLeft, mpORBVocabulary, mK, mDistCoef, mbf, mThDepth, mpCamera);
        else if (mSensor == System::IMU_RGBD)
            mCurrentFrame = Frame(mImGray, imDepth, timestamp, mpORBextractorLeft, mpORBVocabulary, mK, mDistCoef, mbf, mThDepth, mpCamera, &mLastFrame, *mpImuCalib);

        if (ydm_pipe_profile)
        {
            const auto t1 = std::chrono::steady_clock::now();
            ydm_ms_frame = ydm_pipe_ms_since(ydm_pipe_t, t1);
            ydm_pipe_t = t1;
        }

        // DIO-SLAM-style pruning is applied inside TrackWithMotionModel() after pose prediction.

        mCurrentFrame.mNameFile = filename;
        mCurrentFrame.mnDataset = mnNumDataset;

#ifdef REGISTER_TIMES
        vdORBExtract_ms.push_back(mCurrentFrame.mTimeORB_Ext);
#endif
        // 保存YOLO masks和areas到临时变量，供CheckFlowConsistency使用
        // 注意：不要清空mpORBextractorLeft的masks，因为CheckFlowConsistency需要它们
        mpDetector->mvDynamicArea.clear();
        mpDetector->mmDetectMap.clear();
        mpDetector->mask.release();
        mpDetector->mvDynamicMask.clear();

        Track();

        if (ydm_pipe_profile)
        {
            const auto t1 = std::chrono::steady_clock::now();
            ydm_ms_track = ydm_pipe_ms_since(ydm_pipe_t, t1);
            ydm_pipe_t = t1;
        }

        if (ydm_pipe_profile)
        {
            const auto t1 = std::chrono::steady_clock::now();
            ydm_ms_cleanup = ydm_pipe_ms_since(ydm_pipe_t, t1);
            const double ydm_ms_total = ydm_pipe_ms_since(ydm_pipe_t0, t1);
            if ((mCurrentFrame.mnId % ydm_pipe_every) == 0)
            {
                std::cout << "[YDM][PIPE][RGBD] F=" << std::setw(6) << mCurrentFrame.mnId
                          << " Prev=" << std::setw(6) << std::fixed << std::setprecision(2) << ydm_ms_prev << "ms"
                          << " YOLO=" << std::setw(6) << ydm_ms_yolo << "ms"
                          << " Gray=" << std::setw(6) << ydm_ms_gray << "ms"
                          << " Depth=" << std::setw(6) << ydm_ms_depth << "ms"
                          << " Frame=" << std::setw(6) << ydm_ms_frame << "ms"
                          << " Track=" << std::setw(6) << ydm_ms_track << "ms"
                          << " || TOTAL=" << std::setw(7) << ydm_ms_total << "ms"
                          << " (every=" << ydm_pipe_every << ")"
                          << std::endl;
            }
        }

        return mCurrentFrame.GetPose();
    }

    Sophus::SE3f Tracking::GrabImageMonocular(const cv::Mat &im, const double &timestamp, string filename)
    {
        // Store previous frame image for Optical Flow
        if(!mImGray.empty())
            mLastImGray = mImGray.clone();

        mImGray = im;
        if (mImGray.channels() == 3)
        {
            if (mbRGB)
                cvtColor(mImGray, mImGray, cv::COLOR_RGB2GRAY);
            else
                cvtColor(mImGray, mImGray, cv::COLOR_BGR2GRAY);
        }
        else if (mImGray.channels() == 4)
        {
            if (mbRGB)
                cvtColor(mImGray, mImGray, cv::COLOR_RGBA2GRAY);
            else
                cvtColor(mImGray, mImGray, cv::COLOR_BGRA2GRAY);
        }

        if (mSensor == System::MONOCULAR)
        {
            if (mState == NOT_INITIALIZED || mState == NO_IMAGES_YET || (lastID - initID) < mMaxFrames)
                mCurrentFrame = Frame(mImGray, timestamp, mpIniORBextractor, mpORBVocabulary, mpCamera, mDistCoef, mbf, mThDepth);
            else
                mCurrentFrame = Frame(mImGray, timestamp, mpORBextractorLeft, mpORBVocabulary, mpCamera, mDistCoef, mbf, mThDepth);
        }
        else if (mSensor == System::IMU_MONOCULAR)
        {
            if (mState == NOT_INITIALIZED || mState == NO_IMAGES_YET)
            {
                mCurrentFrame = Frame(mImGray, timestamp, mpIniORBextractor, mpORBVocabulary, mpCamera, mDistCoef, mbf, mThDepth, &mLastFrame, *mpImuCalib);
            }
            else
            {
                mCurrentFrame = Frame(mImGray, timestamp, mpORBextractorLeft, mpORBVocabulary, mpCamera, mDistCoef, mbf, mThDepth, &mLastFrame, *mpImuCalib);
            }
        }

        // RDFP-SLAM: Check Optical Flow Consistency immediately after Frame creation
        if(!mLastImGray.empty())
            CheckFlowConsistency();

        if (mState == NO_IMAGES_YET)
            t0 = timestamp;

        mCurrentFrame.mNameFile = filename;
        mCurrentFrame.mnDataset = mnNumDataset;

#ifdef REGISTER_TIMES
        vdORBExtract_ms.push_back(mCurrentFrame.mTimeORB_Ext);
#endif

        lastID = mCurrentFrame.mnId;
        Track();

        return mCurrentFrame.GetPose();
    }

    void Tracking::GrabImuData(const IMU::Point &imuMeasurement)
    {
        unique_lock<mutex> lock(mMutexImuQueue);
        mlQueueImuData.push_back(imuMeasurement);
    }

    void Tracking::PreintegrateIMU()
    {

        if (!mCurrentFrame.mpPrevFrame)
        {
            Verbose::PrintMess("non prev frame ", Verbose::VERBOSITY_NORMAL);
            mCurrentFrame.setIntegrated();
            return;
        }

        mvImuFromLastFrame.clear();
        mvImuFromLastFrame.reserve(mlQueueImuData.size());
        if (mlQueueImuData.size() == 0)
        {
            Verbose::PrintMess("Not IMU data in mlQueueImuData!!", Verbose::VERBOSITY_NORMAL);
            mCurrentFrame.setIntegrated();
            return;
        }

        while (true)
        {
            bool bSleep = false;
            {
                unique_lock<mutex> lock(mMutexImuQueue);
                if (!mlQueueImuData.empty())
                {
                    IMU::Point *m = &mlQueueImuData.front();
                    cout.precision(17);
                    if (m->t < mCurrentFrame.mpPrevFrame->mTimeStamp - mImuPer)
                    {
                        mlQueueImuData.pop_front();
                    }
                    else if (m->t < mCurrentFrame.mTimeStamp - mImuPer)
                    {
                        mvImuFromLastFrame.push_back(*m);
                        mlQueueImuData.pop_front();
                    }
                    else
                    {
                        mvImuFromLastFrame.push_back(*m);
                        break;
                    }
                }
                else
                {
                    break;
                    bSleep = true;
                }
            }
            if (bSleep)
                usleep(500);
        }

        const int n = mvImuFromLastFrame.size() - 1;
        if (n == 0)
        {
            cout << "Empty IMU measurements vector!!!\n";
            return;
        }

        IMU::Preintegrated *pImuPreintegratedFromLastFrame = new IMU::Preintegrated(mLastFrame.mImuBias, mCurrentFrame.mImuCalib);

        for (int i = 0; i < n; i++)
        {
            float tstep;
            Eigen::Vector3f acc, angVel;
            if ((i == 0) && (i < (n - 1)))
            {
                float tab = mvImuFromLastFrame[i + 1].t - mvImuFromLastFrame[i].t;
                float tini = mvImuFromLastFrame[i].t - mCurrentFrame.mpPrevFrame->mTimeStamp;
                acc = (mvImuFromLastFrame[i].a + mvImuFromLastFrame[i + 1].a -
                       (mvImuFromLastFrame[i + 1].a - mvImuFromLastFrame[i].a) * (tini / tab)) *
                      0.5f;
                angVel = (mvImuFromLastFrame[i].w + mvImuFromLastFrame[i + 1].w -
                          (mvImuFromLastFrame[i + 1].w - mvImuFromLastFrame[i].w) * (tini / tab)) *
                         0.5f;
                tstep = mvImuFromLastFrame[i + 1].t - mCurrentFrame.mpPrevFrame->mTimeStamp;
            }
            else if (i < (n - 1))
            {
                acc = (mvImuFromLastFrame[i].a + mvImuFromLastFrame[i + 1].a) * 0.5f;
                angVel = (mvImuFromLastFrame[i].w + mvImuFromLastFrame[i + 1].w) * 0.5f;
                tstep = mvImuFromLastFrame[i + 1].t - mvImuFromLastFrame[i].t;
            }
            else if ((i > 0) && (i == (n - 1)))
            {
                float tab = mvImuFromLastFrame[i + 1].t - mvImuFromLastFrame[i].t;
                float tend = mvImuFromLastFrame[i + 1].t - mCurrentFrame.mTimeStamp;
                acc = (mvImuFromLastFrame[i].a + mvImuFromLastFrame[i + 1].a -
                       (mvImuFromLastFrame[i + 1].a - mvImuFromLastFrame[i].a) * (tend / tab)) *
                      0.5f;
                angVel = (mvImuFromLastFrame[i].w + mvImuFromLastFrame[i + 1].w -
                          (mvImuFromLastFrame[i + 1].w - mvImuFromLastFrame[i].w) * (tend / tab)) *
                         0.5f;
                tstep = mCurrentFrame.mTimeStamp - mvImuFromLastFrame[i].t;
            }
            else if ((i == 0) && (i == (n - 1)))
            {
                acc = mvImuFromLastFrame[i].a;
                angVel = mvImuFromLastFrame[i].w;
                tstep = mCurrentFrame.mTimeStamp - mCurrentFrame.mpPrevFrame->mTimeStamp;
            }

            if (!mpImuPreintegratedFromLastKF)
                cout << "mpImuPreintegratedFromLastKF does not exist" << endl;
            mpImuPreintegratedFromLastKF->IntegrateNewMeasurement(acc, angVel, tstep);
            pImuPreintegratedFromLastFrame->IntegrateNewMeasurement(acc, angVel, tstep);
        }

        mCurrentFrame.mpImuPreintegratedFrame = pImuPreintegratedFromLastFrame;
        mCurrentFrame.mpImuPreintegrated = mpImuPreintegratedFromLastKF;
        mCurrentFrame.mpLastKeyFrame = mpLastKeyFrame;

        mCurrentFrame.setIntegrated();

        // Verbose::PrintMess("Preintegration is finished!! ", Verbose::VERBOSITY_DEBUG);
    }

    bool Tracking::PredictStateIMU()
    {
        if (!mCurrentFrame.mpPrevFrame)
        {
            Verbose::PrintMess("No last frame", Verbose::VERBOSITY_NORMAL);
            return false;
        }

        if (mbMapUpdated && mpLastKeyFrame)
        {
            const Eigen::Vector3f twb1 = mpLastKeyFrame->GetImuPosition();
            const Eigen::Matrix3f Rwb1 = mpLastKeyFrame->GetImuRotation();
            const Eigen::Vector3f Vwb1 = mpLastKeyFrame->GetVelocity();

            const Eigen::Vector3f Gz(0, 0, -IMU::GRAVITY_VALUE);
            const float t12 = mpImuPreintegratedFromLastKF->dT;

            Eigen::Matrix3f Rwb2 = IMU::NormalizeRotation(Rwb1 * mpImuPreintegratedFromLastKF->GetDeltaRotation(mpLastKeyFrame->GetImuBias()));
            Eigen::Vector3f twb2 = twb1 + Vwb1 * t12 + 0.5f * t12 * t12 * Gz + Rwb1 * mpImuPreintegratedFromLastKF->GetDeltaPosition(mpLastKeyFrame->GetImuBias());
            Eigen::Vector3f Vwb2 = Vwb1 + t12 * Gz + Rwb1 * mpImuPreintegratedFromLastKF->GetDeltaVelocity(mpLastKeyFrame->GetImuBias());
            mCurrentFrame.SetImuPoseVelocity(Rwb2, twb2, Vwb2);

            mCurrentFrame.mImuBias = mpLastKeyFrame->GetImuBias();
            mCurrentFrame.mPredBias = mCurrentFrame.mImuBias;
            return true;
        }
        else if (!mbMapUpdated)
        {
            const Eigen::Vector3f twb1 = mLastFrame.GetImuPosition();
            const Eigen::Matrix3f Rwb1 = mLastFrame.GetImuRotation();
            const Eigen::Vector3f Vwb1 = mLastFrame.GetVelocity();
            const Eigen::Vector3f Gz(0, 0, -IMU::GRAVITY_VALUE);
            const float t12 = mCurrentFrame.mpImuPreintegratedFrame->dT;

            Eigen::Matrix3f Rwb2 = IMU::NormalizeRotation(Rwb1 * mCurrentFrame.mpImuPreintegratedFrame->GetDeltaRotation(mLastFrame.mImuBias));
            Eigen::Vector3f twb2 = twb1 + Vwb1 * t12 + 0.5f * t12 * t12 * Gz + Rwb1 * mCurrentFrame.mpImuPreintegratedFrame->GetDeltaPosition(mLastFrame.mImuBias);
            Eigen::Vector3f Vwb2 = Vwb1 + t12 * Gz + Rwb1 * mCurrentFrame.mpImuPreintegratedFrame->GetDeltaVelocity(mLastFrame.mImuBias);

            mCurrentFrame.SetImuPoseVelocity(Rwb2, twb2, Vwb2);

            mCurrentFrame.mImuBias = mLastFrame.mImuBias;
            mCurrentFrame.mPredBias = mCurrentFrame.mImuBias;
            return true;
        }
        else
            cout << "not IMU prediction!!" << endl;

        return false;
    }

    void Tracking::ResetFrameIMU()
    {
        // TODO To implement...
    }

    void Tracking::Track()
    {

        // End-to-end pipeline profiling (core tracking stages)
        const bool ydm_pipe_profile = (std::getenv("YDM_PIPELINE_PROFILE") && std::string(std::getenv("YDM_PIPELINE_PROFILE")) == "1");
        int ydm_pipe_every = 1;
        if (const char* env_pe = std::getenv("YDM_PIPELINE_PROFILE_EVERY"))
        {
            try { ydm_pipe_every = std::stoi(std::string(env_pe)); } catch (...) { ydm_pipe_every = 1; }
            if (ydm_pipe_every < 1) ydm_pipe_every = 1;
        }
        std::chrono::steady_clock::time_point ydm_pipe_t0;
        auto ydm_pipe_ms_since = [](const std::chrono::steady_clock::time_point &a,
                                    const std::chrono::steady_clock::time_point &b) -> double {
            return std::chrono::duration_cast<std::chrono::duration<double, std::milli>>(b - a).count();
        };
        double ydm_ms_preimu = 0.0, ydm_ms_init = 0.0, ydm_ms_pose = 0.0;
        double ydm_ms_local = 0.0, ydm_ms_draw = 0.0, ydm_ms_kf = 0.0;
        if (ydm_pipe_profile) ydm_pipe_t0 = std::chrono::steady_clock::now();

        if (bStepByStep)
        {
            std::cout << "Tracking: Waiting to the next step" << std::endl;
            while (!mbStep && bStepByStep)
                usleep(500);
            mbStep = false;
        }

        if (mpLocalMapper->mbBadImu)
        {
            cout << "TRACK: Reset map because local mapper set the bad imu flag " << endl;
            mpSystem->ResetActiveMap();
            return;
        }

        Map *pCurrentMap = mpAtlas->GetCurrentMap();
        if (!pCurrentMap)
        {
            cout << "ERROR: There is not an active map in the atlas" << endl;
        }

        if (mState != NO_IMAGES_YET)
        {
            if (mLastFrame.mTimeStamp > mCurrentFrame.mTimeStamp)
            {
                cerr << "ERROR: Frame with a timestamp older than previous frame detected!" << endl;
                unique_lock<mutex> lock(mMutexImuQueue);
                mlQueueImuData.clear();
                CreateMapInAtlas();
                return;
            }
            else if (mCurrentFrame.mTimeStamp > mLastFrame.mTimeStamp + 1.0)
            {
                // cout << mCurrentFrame.mTimeStamp << ", " << mLastFrame.mTimeStamp << endl;
                // cout << "id last: " << mLastFrame.mnId << "    id curr: " << mCurrentFrame.mnId << endl;
                if (mpAtlas->isInertial())
                {

                    if (mpAtlas->isImuInitialized())
                    {
                        cout << "Timestamp jump detected. State set to LOST. Reseting IMU integration..." << endl;
                        if (!pCurrentMap->GetIniertialBA2())
                        {
                            mpSystem->ResetActiveMap();
                        }
                        else
                        {
                            CreateMapInAtlas();
                        }
                    }
                    else
                    {
                        cout << "Timestamp jump detected, before IMU initialization. Reseting..." << endl;
                        mpSystem->ResetActiveMap();
                    }
                    return;
                }
            }
        }

        if ((mSensor == System::IMU_MONOCULAR || mSensor == System::IMU_STEREO || mSensor == System::IMU_RGBD) && mpLastKeyFrame)
            mCurrentFrame.SetNewBias(mpLastKeyFrame->GetImuBias());

        if (mState == NO_IMAGES_YET)
        {
            mState = NOT_INITIALIZED;
        }

        mLastProcessedState = mState;

        if ((mSensor == System::IMU_MONOCULAR || mSensor == System::IMU_STEREO || mSensor == System::IMU_RGBD) && !mbCreatedMap)
        {
    #ifndef REGISTER_TIMES
            std::chrono::steady_clock::time_point time_StartPreIMU;
            if (ydm_pipe_profile) time_StartPreIMU = std::chrono::steady_clock::now();
    #endif
#ifdef REGISTER_TIMES
            std::chrono::steady_clock::time_point time_StartPreIMU = std::chrono::steady_clock::now();
#endif
            PreintegrateIMU();
#ifdef REGISTER_TIMES
            std::chrono::steady_clock::time_point time_EndPreIMU = std::chrono::steady_clock::now();

            double timePreImu = std::chrono::duration_cast<std::chrono::duration<double, std::milli>>(time_EndPreIMU - time_StartPreIMU).count();
            vdIMUInteg_ms.push_back(timePreImu);
#endif

            if (ydm_pipe_profile)
            {
                const auto time_EndPreIMU2 = std::chrono::steady_clock::now();
                ydm_ms_preimu = ydm_pipe_ms_since(time_StartPreIMU, time_EndPreIMU2);
            }
        }
        mbCreatedMap = false;

        // Get Map Mutex -> Map cannot be changed
        unique_lock<mutex> lock(pCurrentMap->mMutexMapUpdate);

        mbMapUpdated = false;

        int nCurMapChangeIndex = pCurrentMap->GetMapChangeIndex();
        int nMapChangeIndex = pCurrentMap->GetLastMapChange();
        if (nCurMapChangeIndex > nMapChangeIndex)
        {
            pCurrentMap->SetLastMapChange(nCurMapChangeIndex);
            mbMapUpdated = true;
        }

        if (mState == NOT_INITIALIZED)
        {
            std::chrono::steady_clock::time_point t_init0;
            if (ydm_pipe_profile) t_init0 = std::chrono::steady_clock::now();
            if (mSensor == System::STEREO || mSensor == System::RGBD || mSensor == System::IMU_STEREO || mSensor == System::IMU_RGBD)
            {
                StereoInitialization();
            }
            else
            {
                MonocularInitialization();
            }

            if (ydm_pipe_profile)
            {
                const auto t_init1 = std::chrono::steady_clock::now();
                ydm_ms_init = ydm_pipe_ms_since(t_init0, t_init1);
            }

            // mpFrameDrawer->Update(this);

            if (mState != OK) // If rightly initialized, mState=OK
            {
                mLastFrame = Frame(mCurrentFrame);

                if (ydm_pipe_profile && (mCurrentFrame.mnId % ydm_pipe_every) == 0)
                {
                    const auto t_end = std::chrono::steady_clock::now();
                    const double ms_total = ydm_pipe_ms_since(ydm_pipe_t0, t_end);
                    std::cout << "[YDM][PIPE][TRACK] F=" << std::setw(6) << mCurrentFrame.mnId
                              << " PreIMU=" << std::setw(6) << std::fixed << std::setprecision(2) << ydm_ms_preimu << "ms"
                              << " Init=" << std::setw(6) << ydm_ms_init << "ms"
                              << " || TOTAL=" << std::setw(7) << ms_total << "ms"
                              << " (every=" << ydm_pipe_every << ")"
                              << std::endl;
                }
                return;
            }

            if (mpAtlas->GetAllMaps().size() == 1)
            {
                mnFirstFrameId = mCurrentFrame.mnId;
            }
        }
        else
        {
            // System is initialized. Track Frame.
            bool bOK;

            std::chrono::steady_clock::time_point t_pose0;
            if (ydm_pipe_profile) t_pose0 = std::chrono::steady_clock::now();

#ifdef REGISTER_TIMES
            std::chrono::steady_clock::time_point time_StartPosePred = std::chrono::steady_clock::now();
#endif

            // Initial camera pose estimation using motion model or relocalization (if tracking is lost)
            if (!mbOnlyTracking)
            {

                // State OK
                // Local Mapping is activated. This is the normal behaviour, unless
                // you explicitly activate the "only tracking" mode.
                if (mState == OK)
                {

                    // Local Mapping might have changed some MapPoints tracked in last frame
                    CheckReplacedInLastFrame();

                    if ((!mbVelocity && !pCurrentMap->isImuInitialized()) || mCurrentFrame.mnId < mnLastRelocFrameId + 2)
                    {
                        Verbose::PrintMess("TRACK: Track with respect to the reference KF ", Verbose::VERBOSITY_DEBUG);
                        bOK = TrackReferenceKeyFrame();
                    }
                    else
                    {
                        Verbose::PrintMess("TRACK: Track with motion model", Verbose::VERBOSITY_DEBUG);
                        bOK = TrackWithMotionModel();
                        if (!bOK)
                            bOK = TrackReferenceKeyFrame();
                    }

                    if (!bOK)
                    {
                        if (mCurrentFrame.mnId <= (mnLastRelocFrameId + mnFramesToResetIMU) &&
                            (mSensor == System::IMU_MONOCULAR || mSensor == System::IMU_STEREO || mSensor == System::IMU_RGBD))
                        {
                            mState = LOST;
                        }
                        else if (pCurrentMap->KeyFramesInMap() > 10)
                        {
                            // cout << "KF in map: " << pCurrentMap->KeyFramesInMap() << endl;
                            mState = RECENTLY_LOST;
                            mTimeStampLost = mCurrentFrame.mTimeStamp;
                        }
                        else
                        {
                            mState = LOST;
                        }
                    }
                }
                else
                {

                    if (mState == RECENTLY_LOST)
                    {
                        Verbose::PrintMess("Lost for a short time", Verbose::VERBOSITY_NORMAL);

                        bOK = true;
                        if ((mSensor == System::IMU_MONOCULAR || mSensor == System::IMU_STEREO || mSensor == System::IMU_RGBD))
                        {
                            if (pCurrentMap->isImuInitialized())
                                PredictStateIMU();
                            else
                                bOK = false;

                            if (mCurrentFrame.mTimeStamp - mTimeStampLost > time_recently_lost)
                            {
                                mState = LOST;
                                Verbose::PrintMess("Track Lost...", Verbose::VERBOSITY_NORMAL);
                                bOK = false;
                            }
                        }
                        else
                        {
                            // Relocalization
                            bOK = Relocalization();
                            // std::cout << "mCurrentFrame.mTimeStamp:" << to_string(mCurrentFrame.mTimeStamp) << std::endl;
                            // std::cout << "mTimeStampLost:" << to_string(mTimeStampLost) << std::endl;
                            if (mCurrentFrame.mTimeStamp - mTimeStampLost > 3.0f && !bOK)
                            {
                                mState = LOST;
                                Verbose::PrintMess("Track Lost...", Verbose::VERBOSITY_NORMAL);
                                bOK = false;
                            }
                        }
                    }
                    else if (mState == LOST)
                    {

                        Verbose::PrintMess("A new map is started...", Verbose::VERBOSITY_NORMAL);

                        if (pCurrentMap->KeyFramesInMap() < 10)
                        {
                            mpSystem->ResetActiveMap();
                            Verbose::PrintMess("Reseting current map...", Verbose::VERBOSITY_NORMAL);
                        }
                        else
                            CreateMapInAtlas();

                        if (mpLastKeyFrame)
                            mpLastKeyFrame = static_cast<KeyFrame *>(NULL);

                        Verbose::PrintMess("done", Verbose::VERBOSITY_NORMAL);

                        return;
                    }
                }
            }
            else
            {
                // Localization Mode: Local Mapping is deactivated (TODO Not available in inertial mode)
                if (mState == LOST)
                {
                    if (mSensor == System::IMU_MONOCULAR || mSensor == System::IMU_STEREO || mSensor == System::IMU_RGBD)
                        Verbose::PrintMess("IMU. State LOST", Verbose::VERBOSITY_NORMAL);
                    bOK = Relocalization();
                }
                else
                {
                    if (!mbVO)
                    {
                        // In last frame we tracked enough MapPoints in the map
                        if (mbVelocity)
                        {
                            bOK = TrackWithMotionModel();
                        }
                        else
                        {
                            bOK = TrackReferenceKeyFrame();
                        }
                    }
                    else
                    {
                        // In last frame we tracked mainly "visual odometry" points.

                        // We compute two camera poses, one from motion model and one doing relocalization.
                        // If relocalization is sucessfull we choose that solution, otherwise we retain
                        // the "visual odometry" solution.

                        bool bOKMM = false;
                        bool bOKReloc = false;
                        vector<MapPoint *> vpMPsMM;
                        vector<bool> vbOutMM;
                        Sophus::SE3f TcwMM;
                        if (mbVelocity)
                        {
                            bOKMM = TrackWithMotionModel();
                            vpMPsMM = mCurrentFrame.mvpMapPoints;
                            vbOutMM = mCurrentFrame.mvbOutlier;
                            TcwMM = mCurrentFrame.GetPose();
                        }
                        bOKReloc = Relocalization();

                        if (bOKMM && !bOKReloc)
                        {
                            mCurrentFrame.SetPose(TcwMM);
                            mCurrentFrame.mvpMapPoints = vpMPsMM;
                            mCurrentFrame.mvbOutlier = vbOutMM;

                            if (mbVO)
                            {
                                for (int i = 0; i < mCurrentFrame.N; i++)
                                {
                                    if (mCurrentFrame.mvpMapPoints[i] && !mCurrentFrame.mvbOutlier[i])
                                    {
                                        mCurrentFrame.mvpMapPoints[i]->IncreaseFound();
                                    }
                                }
                            }
                        }
                        else if (bOKReloc)
                        {
                            mbVO = false;
                        }

                        bOK = bOKReloc || bOKMM;
                    }
                }
            }

            if (!mCurrentFrame.mpReferenceKF)
                mCurrentFrame.mpReferenceKF = mpReferenceKF;

            if (ydm_pipe_profile)
            {
                const auto t_pose1 = std::chrono::steady_clock::now();
                ydm_ms_pose = ydm_pipe_ms_since(t_pose0, t_pose1);
            }

#ifdef REGISTER_TIMES
            std::chrono::steady_clock::time_point time_EndPosePred = std::chrono::steady_clock::now();

            double timePosePred = std::chrono::duration_cast<std::chrono::duration<double, std::milli>>(time_EndPosePred - time_StartPosePred).count();
            vdPosePred_ms.push_back(timePosePred);
#endif

#ifdef REGISTER_TIMES
            std::chrono::steady_clock::time_point time_StartLMTrack = std::chrono::steady_clock::now();
#endif
            // If we have an initial estimation of the camera pose and matching. Track the local map.
            if (!mbOnlyTracking)
            {
                if (bOK)
                {
                    std::chrono::steady_clock::time_point t_local0;
                    if (ydm_pipe_profile) t_local0 = std::chrono::steady_clock::now();
                    bOK = TrackLocalMap();
                    if (ydm_pipe_profile)
                    {
                        const auto t_local1 = std::chrono::steady_clock::now();
                        ydm_ms_local = ydm_pipe_ms_since(t_local0, t_local1);
                    }
                }
                if (!bOK)
                    cout << "Fail to track local map!" << endl;
            }
            else
            {
                // mbVO true means that there are few matches to MapPoints in the map. We cannot retrieve
                // a local map and therefore we do not perform TrackLocalMap(). Once the system relocalizes
                // the camera we will use the local map again.
                if (bOK && !mbVO)
                {
                    std::chrono::steady_clock::time_point t_local0;
                    if (ydm_pipe_profile) t_local0 = std::chrono::steady_clock::now();
                    bOK = TrackLocalMap();
                    if (ydm_pipe_profile)
                    {
                        const auto t_local1 = std::chrono::steady_clock::now();
                        ydm_ms_local = ydm_pipe_ms_since(t_local0, t_local1);
                    }
                }
            }

            if (bOK)
            {
                // RGB-D/IMU_RGBD：动态剔除与mask refinement已在 GrabImageRGBD() 中通过
                // VerifyAndPruneFeatures() 完成。这里再调用 CheckFlowConsistency()
                // 会用 YOLO 原始 mask（并可能膨胀）覆盖 mCurrentFrame.mvDynamicMask，
                // 导致点云(PCL)侧仍按 YOLO mask 把“静止的人/物体”剔除。
                if (mSensor != System::RGBD && mSensor != System::IMU_RGBD)
                {
                    CheckFlowConsistency();
                }
                mState = OK;
            }
            else if (mState == OK)
            {
                if (mSensor == System::IMU_MONOCULAR || mSensor == System::IMU_STEREO || mSensor == System::IMU_RGBD)
                {
                    Verbose::PrintMess("Track lost for less than one second...", Verbose::VERBOSITY_NORMAL);
                    if (!pCurrentMap->isImuInitialized() || !pCurrentMap->GetIniertialBA2())
                    {
                        cout << "IMU is not or recently initialized. Reseting active map..." << endl;
                        mpSystem->ResetActiveMap();
                    }

                    mState = RECENTLY_LOST;
                }
                else
                    mState = RECENTLY_LOST; // visual to lost

                /*if(mCurrentFrame.mnId>mnLastRelocFrameId+mMaxFrames)
                {*/
                mTimeStampLost = mCurrentFrame.mTimeStamp;
                //}
            }

            // Save frame if recent relocalization, since they are used for IMU reset (as we are making copy, it shluld be once mCurrFrame is completely modified)
            if ((mCurrentFrame.mnId < (mnLastRelocFrameId + mnFramesToResetIMU)) && (mCurrentFrame.mnId > mnFramesToResetIMU) &&
                (mSensor == System::IMU_MONOCULAR || mSensor == System::IMU_STEREO || mSensor == System::IMU_RGBD) && pCurrentMap->isImuInitialized())
            {
                // TODO check this situation
                Verbose::PrintMess("Saving pointer to frame. imu needs reset...", Verbose::VERBOSITY_NORMAL);
                Frame *pF = new Frame(mCurrentFrame);
                pF->mpPrevFrame = new Frame(mLastFrame);

                // Load preintegration
                pF->mpImuPreintegratedFrame = new IMU::Preintegrated(mCurrentFrame.mpImuPreintegratedFrame);
            }

            if (pCurrentMap->isImuInitialized())
            {
                if (bOK)
                {
                    if (mCurrentFrame.mnId == (mnLastRelocFrameId + mnFramesToResetIMU))
                    {
                        cout << "RESETING FRAME!!!" << endl;
                        ResetFrameIMU();
                    }
                    else if (mCurrentFrame.mnId > (mnLastRelocFrameId + 30))
                        mLastBias = mCurrentFrame.mImuBias;
                }
            }

#ifdef REGISTER_TIMES
            std::chrono::steady_clock::time_point time_EndLMTrack = std::chrono::steady_clock::now();

            double timeLMTrack = std::chrono::duration_cast<std::chrono::duration<double, std::milli>>(time_EndLMTrack - time_StartLMTrack).count();
            vdLMTrack_ms.push_back(timeLMTrack);
#endif

            // Update drawer
            std::chrono::steady_clock::time_point t_draw0;
            if (ydm_pipe_profile) t_draw0 = std::chrono::steady_clock::now();
            mpFrameDrawer->Update(this);
            if (mCurrentFrame.isSet())
                mpMapDrawer->SetCurrentCameraPose(mCurrentFrame.GetPose());
            if (ydm_pipe_profile)
            {
                const auto t_draw1 = std::chrono::steady_clock::now();
                ydm_ms_draw = ydm_pipe_ms_since(t_draw0, t_draw1);
            }

            if (bOK || mState == RECENTLY_LOST)
            {
                // Update motion model
                if (mLastFrame.isSet() && mCurrentFrame.isSet())
                {
                    Sophus::SE3f LastTwc = mLastFrame.GetPose().inverse();
                    mVelocity = mCurrentFrame.GetPose() * LastTwc;
                    mbVelocity = true;
                }
                else
                {
                    mbVelocity = false;
                }

                if (mSensor == System::IMU_MONOCULAR || mSensor == System::IMU_STEREO || mSensor == System::IMU_RGBD)
                    mpMapDrawer->SetCurrentCameraPose(mCurrentFrame.GetPose());

                // Clean VO matches
                for (int i = 0; i < mCurrentFrame.N; i++)
                {
                    MapPoint *pMP = mCurrentFrame.mvpMapPoints[i];
                    if (pMP)
                        if (pMP->Observations() < 1)
                        {
                            mCurrentFrame.mvbOutlier[i] = false;
                            mCurrentFrame.mvpMapPoints[i] = static_cast<MapPoint *>(NULL);
                        }
                }

                // Delete temporal MapPoints
                for (list<MapPoint *>::iterator lit = mlpTemporalPoints.begin(), lend = mlpTemporalPoints.end(); lit != lend; lit++)
                {
                    MapPoint *pMP = *lit;
                    delete pMP;
                }
                mlpTemporalPoints.clear();

#ifdef REGISTER_TIMES
                std::chrono::steady_clock::time_point time_StartNewKF = std::chrono::steady_clock::now();
#endif
                std::chrono::steady_clock::time_point t_kf0;
                if (ydm_pipe_profile) t_kf0 = std::chrono::steady_clock::now();
                bool bNeedKF = NeedNewKeyFrame();

                // Check if we need to insert a new keyframe
                // if(bNeedKF && bOK)
                if (bNeedKF && (bOK || (mInsertKFsLost && mState == RECENTLY_LOST &&
                                        (mSensor == System::IMU_MONOCULAR || mSensor == System::IMU_STEREO || mSensor == System::IMU_RGBD))))
                    CreateNewKeyFrame();

                if (ydm_pipe_profile)
                {
                    const auto t_kf1 = std::chrono::steady_clock::now();
                    ydm_ms_kf = ydm_pipe_ms_since(t_kf0, t_kf1);
                }

#ifdef REGISTER_TIMES
                std::chrono::steady_clock::time_point time_EndNewKF = std::chrono::steady_clock::now();

                double timeNewKF = std::chrono::duration_cast<std::chrono::duration<double, std::milli>>(time_EndNewKF - time_StartNewKF).count();
                vdNewKF_ms.push_back(timeNewKF);
#endif

                // We allow points with high innovation (considererd outliers by the Huber Function)
                // pass to the new keyframe, so that bundle adjustment will finally decide
                // if they are outliers or not. We don't want next frame to estimate its position
                // with those points so we discard them in the frame. Only has effect if lastframe is tracked
                for (int i = 0; i < mCurrentFrame.N; i++)
                {
                    if (mCurrentFrame.mvpMapPoints[i] && mCurrentFrame.mvbOutlier[i])
                        mCurrentFrame.mvpMapPoints[i] = static_cast<MapPoint *>(NULL);
                }
            }

            // Reset if the camera get lost soon after initialization
            if (mState == LOST)
            {
                if (pCurrentMap->KeyFramesInMap() <= 10)
                {
                    mpSystem->ResetActiveMap();
                    return;
                }
                if (mSensor == System::IMU_MONOCULAR || mSensor == System::IMU_STEREO || mSensor == System::IMU_RGBD)
                    if (!pCurrentMap->isImuInitialized())
                    {
                        Verbose::PrintMess("Track lost before IMU initialisation, reseting...", Verbose::VERBOSITY_QUIET);
                        mpSystem->ResetActiveMap();
                        return;
                    }

                CreateMapInAtlas();

                return;
            }

            if (!mCurrentFrame.mpReferenceKF)
                mCurrentFrame.mpReferenceKF = mpReferenceKF;

            mLastFrame = Frame(mCurrentFrame);
        }

        if (ydm_pipe_profile && (mCurrentFrame.mnId % ydm_pipe_every) == 0)
        {
            const auto t_end = std::chrono::steady_clock::now();
            const double ms_total = ydm_pipe_ms_since(ydm_pipe_t0, t_end);
            const double ms_other = ms_total - (ydm_ms_preimu + ydm_ms_init + ydm_ms_pose + ydm_ms_local + ydm_ms_draw + ydm_ms_kf);
            std::cout << "[YDM][PIPE][TRACK] F=" << std::setw(6) << mCurrentFrame.mnId
                      << " PreIMU=" << std::setw(6) << std::fixed << std::setprecision(2) << ydm_ms_preimu << "ms"
                      << " Pose=" << std::setw(6) << ydm_ms_pose << "ms"
                      << " Local=" << std::setw(6) << ydm_ms_local << "ms"
                      << " Draw=" << std::setw(6) << ydm_ms_draw << "ms"
                      << " KF=" << std::setw(6) << ydm_ms_kf << "ms"
                      << " Other=" << std::setw(6) << ms_other << "ms"
                      << " || TOTAL=" << std::setw(7) << ms_total << "ms"
                      << " (every=" << ydm_pipe_every << ")"
                      << std::endl;
        }

        if (mState == OK || mState == RECENTLY_LOST)
        {
            // Store frame pose information to retrieve the complete camera trajectory afterwards.
            if (mCurrentFrame.isSet())
            {
                Sophus::SE3f Tcr_ = mCurrentFrame.GetPose() * mCurrentFrame.mpReferenceKF->GetPoseInverse();
                mlRelativeFramePoses.push_back(Tcr_);
                mlpReferences.push_back(mCurrentFrame.mpReferenceKF);
                mlFrameTimes.push_back(mCurrentFrame.mTimeStamp);
                mlbLost.push_back(mState == LOST);
            }
            else
            {
                // This can happen if tracking is lost
                mlRelativeFramePoses.push_back(mlRelativeFramePoses.back());
                mlpReferences.push_back(mlpReferences.back());
                mlFrameTimes.push_back(mlFrameTimes.back());
                mlbLost.push_back(mState == LOST);
            }
        }

#ifdef REGISTER_LOOP
        if (Stop())
        {

            // Safe area to stop
            while (isStopped())
            {
                usleep(3000);
            }
        }
#endif
    }

    void Tracking::StereoInitialization()
    {
        if (mCurrentFrame.N > 500)
        {
            if (mSensor == System::IMU_STEREO || mSensor == System::IMU_RGBD)
            {
                if (!mCurrentFrame.mpImuPreintegrated || !mLastFrame.mpImuPreintegrated)
                {
                    cout << "not IMU meas" << endl;
                    return;
                }

                if (!mFastInit && (mCurrentFrame.mpImuPreintegratedFrame->avgA - mLastFrame.mpImuPreintegratedFrame->avgA).norm() < 0.5)
                {
                    cout << "not enough acceleration" << endl;
                    return;
                }

                if (mpImuPreintegratedFromLastKF)
                    delete mpImuPreintegratedFromLastKF;

                mpImuPreintegratedFromLastKF = new IMU::Preintegrated(IMU::Bias(), *mpImuCalib);
                mCurrentFrame.mpImuPreintegrated = mpImuPreintegratedFromLastKF;
            }

            // Set Frame pose to the origin (In case of inertial SLAM to imu)
            if (mSensor == System::IMU_STEREO || mSensor == System::IMU_RGBD)
            {
                Eigen::Matrix3f Rwb0 = mCurrentFrame.mImuCalib.mTcb.rotationMatrix();
                Eigen::Vector3f twb0 = mCurrentFrame.mImuCalib.mTcb.translation();
                Eigen::Vector3f Vwb0;
                Vwb0.setZero();
                mCurrentFrame.SetImuPoseVelocity(Rwb0, twb0, Vwb0);
            }
            else
                mCurrentFrame.SetPose(Sophus::SE3f());

            // Create KeyFrame
            KeyFrame *pKFini = new KeyFrame(mCurrentFrame, mpAtlas->GetCurrentMap(), mpKeyFrameDB);

            // Insert KeyFrame in the map
            mpAtlas->AddKeyFrame(pKFini);

            // Create MapPoints and asscoiate to KeyFrame
            if (!mpCamera2)
            {
                for (int i = 0; i < mCurrentFrame.N; i++)
                {
                    float z = mCurrentFrame.mvDepth[i];
                    if (z > 0)
                    {
                        Eigen::Vector3f x3D;
                        mCurrentFrame.UnprojectStereo(i, x3D);
                        MapPoint *pNewMP = new MapPoint(x3D, pKFini, mpAtlas->GetCurrentMap());
                        pNewMP->AddObservation(pKFini, i);
                        pKFini->AddMapPoint(pNewMP, i);
                        pNewMP->ComputeDistinctiveDescriptors();
                        pNewMP->UpdateNormalAndDepth();
                        mpAtlas->AddMapPoint(pNewMP);

                        mCurrentFrame.mvpMapPoints[i] = pNewMP;
                    }
                }
            }
            else
            {
                for (int i = 0; i < mCurrentFrame.Nleft; i++)
                {
                    int rightIndex = mCurrentFrame.mvLeftToRightMatch[i];
                    if (rightIndex != -1)
                    {
                        Eigen::Vector3f x3D = mCurrentFrame.mvStereo3Dpoints[i];

                        MapPoint *pNewMP = new MapPoint(x3D, pKFini, mpAtlas->GetCurrentMap());

                        pNewMP->AddObservation(pKFini, i);
                        pNewMP->AddObservation(pKFini, rightIndex + mCurrentFrame.Nleft);

                        pKFini->AddMapPoint(pNewMP, i);
                        pKFini->AddMapPoint(pNewMP, rightIndex + mCurrentFrame.Nleft);

                        pNewMP->ComputeDistinctiveDescriptors();
                        pNewMP->UpdateNormalAndDepth();
                        mpAtlas->AddMapPoint(pNewMP);

                        mCurrentFrame.mvpMapPoints[i] = pNewMP;
                        mCurrentFrame.mvpMapPoints[rightIndex + mCurrentFrame.Nleft] = pNewMP;
                    }
                }
            }

            Verbose::PrintMess("New Map created with " + to_string(mpAtlas->MapPointsInMap()) + " points", Verbose::VERBOSITY_QUIET);

            // cout << "Active map: " << mpAtlas->GetCurrentMap()->GetId() << endl;

            mpLocalMapper->InsertKeyFrame(pKFini);

            mLastFrame = Frame(mCurrentFrame);
            mnLastKeyFrameId = mCurrentFrame.mnId;
            mpLastKeyFrame = pKFini;
            // mnLastRelocFrameId = mCurrentFrame.mnId;

            mvpLocalKeyFrames.push_back(pKFini);
            mvpLocalMapPoints = mpAtlas->GetAllMapPoints();
            mpReferenceKF = pKFini;
            mCurrentFrame.mpReferenceKF = pKFini;

            mpAtlas->SetReferenceMapPoints(mvpLocalMapPoints);

            mpAtlas->GetCurrentMap()->mvpKeyFrameOrigins.push_back(pKFini);

            mpMapDrawer->SetCurrentCameraPose(mCurrentFrame.GetPose());

            mState = OK;
        }
    }

    void Tracking::MonocularInitialization()
    {

        if (!mbReadyToInitializate)
        {
            // Set Reference Frame
            if (mCurrentFrame.mvKeys.size() > 100)
            {

                mInitialFrame = Frame(mCurrentFrame);
                mLastFrame = Frame(mCurrentFrame);
                mvbPrevMatched.resize(mCurrentFrame.mvKeysUn.size());
                for (size_t i = 0; i < mCurrentFrame.mvKeysUn.size(); i++)
                    mvbPrevMatched[i] = mCurrentFrame.mvKeysUn[i].pt;

                fill(mvIniMatches.begin(), mvIniMatches.end(), -1);

                if (mSensor == System::IMU_MONOCULAR)
                {
                    if (mpImuPreintegratedFromLastKF)
                    {
                        delete mpImuPreintegratedFromLastKF;
                    }
                    mpImuPreintegratedFromLastKF = new IMU::Preintegrated(IMU::Bias(), *mpImuCalib);
                    mCurrentFrame.mpImuPreintegrated = mpImuPreintegratedFromLastKF;
                }

                mbReadyToInitializate = true;

                return;
            }
        }
        else
        {
            if (((int)mCurrentFrame.mvKeys.size() <= 100) || ((mSensor == System::IMU_MONOCULAR) && (mLastFrame.mTimeStamp - mInitialFrame.mTimeStamp > 1.0)))
            {
                mbReadyToInitializate = false;

                return;
            }

            // Find correspondences
            ORBmatcher matcher(0.9, true);
            int nmatches = matcher.SearchForInitialization(mInitialFrame, mCurrentFrame, mvbPrevMatched, mvIniMatches, 100);

            // Check if there are enough correspondences
            if (nmatches < 100)
            {
                mbReadyToInitializate = false;
                return;
            }

            Sophus::SE3f Tcw;
            vector<bool> vbTriangulated; // Triangulated Correspondences (mvIniMatches)

            if (mpCamera->ReconstructWithTwoViews(mInitialFrame.mvKeysUn, mCurrentFrame.mvKeysUn, mvIniMatches, Tcw, mvIniP3D, vbTriangulated))
            {
                for (size_t i = 0, iend = mvIniMatches.size(); i < iend; i++)
                {
                    if (mvIniMatches[i] >= 0 && !vbTriangulated[i])
                    {
                        mvIniMatches[i] = -1;
                        nmatches--;
                    }
                }

                // Set Frame Poses
                mInitialFrame.SetPose(Sophus::SE3f());
                mCurrentFrame.SetPose(Tcw);

                CreateInitialMapMonocular();
            }
        }
    }

    void Tracking::CreateInitialMapMonocular()
    {
        // Create KeyFrames
        KeyFrame *pKFini = new KeyFrame(mInitialFrame, mpAtlas->GetCurrentMap(), mpKeyFrameDB);
        KeyFrame *pKFcur = new KeyFrame(mCurrentFrame, mpAtlas->GetCurrentMap(), mpKeyFrameDB);

        if (mSensor == System::IMU_MONOCULAR)
            pKFini->mpImuPreintegrated = (IMU::Preintegrated *)(NULL);

        pKFini->ComputeBoW();
        pKFcur->ComputeBoW();

        // Insert KFs in the map
        mpAtlas->AddKeyFrame(pKFini);
        mpAtlas->AddKeyFrame(pKFcur);

        for (size_t i = 0; i < mvIniMatches.size(); i++)
        {
            if (mvIniMatches[i] < 0)
                continue;

            // Create MapPoint.
            Eigen::Vector3f worldPos;
            worldPos << mvIniP3D[i].x, mvIniP3D[i].y, mvIniP3D[i].z;
            MapPoint *pMP = new MapPoint(worldPos, pKFcur, mpAtlas->GetCurrentMap());

            pKFini->AddMapPoint(pMP, i);
            pKFcur->AddMapPoint(pMP, mvIniMatches[i]);

            pMP->AddObservation(pKFini, i);
            pMP->AddObservation(pKFcur, mvIniMatches[i]);

            pMP->ComputeDistinctiveDescriptors();
            pMP->UpdateNormalAndDepth();

            // Fill Current Frame structure
            mCurrentFrame.mvpMapPoints[mvIniMatches[i]] = pMP;
            mCurrentFrame.mvbOutlier[mvIniMatches[i]] = false;

            // Add to Map
            mpAtlas->AddMapPoint(pMP);
        }

        // Update Connections
        pKFini->UpdateConnections();
        pKFcur->UpdateConnections();

        std::set<MapPoint *> sMPs;
        sMPs = pKFini->GetMapPoints();

        // Bundle Adjustment
        Verbose::PrintMess("New Map created with " + to_string(mpAtlas->MapPointsInMap()) + " points", Verbose::VERBOSITY_QUIET);
        Optimizer::GlobalBundleAdjustemnt(mpAtlas->GetCurrentMap(), 20);

        float medianDepth = pKFini->ComputeSceneMedianDepth(2);
        float invMedianDepth;
        if (mSensor == System::IMU_MONOCULAR)
            invMedianDepth = 4.0f / medianDepth; // 4.0f
        else
            invMedianDepth = 1.0f / medianDepth;

        if (medianDepth < 0 || pKFcur->TrackedMapPoints(1) < 50) // TODO Check, originally 100 tracks
        {
            Verbose::PrintMess("Wrong initialization, reseting...", Verbose::VERBOSITY_QUIET);
            mpSystem->ResetActiveMap();
            return;
        }

        // Scale initial baseline
        Sophus::SE3f Tc2w = pKFcur->GetPose();
        Tc2w.translation() *= invMedianDepth;
        pKFcur->SetPose(Tc2w);

        // Scale points
        vector<MapPoint *> vpAllMapPoints = pKFini->GetMapPointMatches();
        for (size_t iMP = 0; iMP < vpAllMapPoints.size(); iMP++)
        {
            if (vpAllMapPoints[iMP])
            {
                MapPoint *pMP = vpAllMapPoints[iMP];
                pMP->SetWorldPos(pMP->GetWorldPos() * invMedianDepth);
                pMP->UpdateNormalAndDepth();
            }
        }

        if (mSensor == System::IMU_MONOCULAR)
        {
            pKFcur->mPrevKF = pKFini;
            pKFini->mNextKF = pKFcur;
            pKFcur->mpImuPreintegrated = mpImuPreintegratedFromLastKF;

            mpImuPreintegratedFromLastKF = new IMU::Preintegrated(pKFcur->mpImuPreintegrated->GetUpdatedBias(), pKFcur->mImuCalib);
        }

        mpLocalMapper->InsertKeyFrame(pKFini);
        mpLocalMapper->InsertKeyFrame(pKFcur);
        mpLocalMapper->mFirstTs = pKFcur->mTimeStamp;

        mCurrentFrame.SetPose(pKFcur->GetPose());
        mnLastKeyFrameId = mCurrentFrame.mnId;
        mpLastKeyFrame = pKFcur;
        // mnLastRelocFrameId = mInitialFrame.mnId;

        mvpLocalKeyFrames.push_back(pKFcur);
        mvpLocalKeyFrames.push_back(pKFini);
        mvpLocalMapPoints = mpAtlas->GetAllMapPoints();
        mpReferenceKF = pKFcur;
        mCurrentFrame.mpReferenceKF = pKFcur;

        // Compute here initial velocity
        vector<KeyFrame *> vKFs = mpAtlas->GetAllKeyFrames();

        Sophus::SE3f deltaT = vKFs.back()->GetPose() * vKFs.front()->GetPoseInverse();
        mbVelocity = false;
        Eigen::Vector3f phi = deltaT.so3().log();

        double aux = (mCurrentFrame.mTimeStamp - mLastFrame.mTimeStamp) / (mCurrentFrame.mTimeStamp - mInitialFrame.mTimeStamp);
        phi *= aux;

        mLastFrame = Frame(mCurrentFrame);

        mpAtlas->SetReferenceMapPoints(mvpLocalMapPoints);

        mpMapDrawer->SetCurrentCameraPose(pKFcur->GetPose());

        mpAtlas->GetCurrentMap()->mvpKeyFrameOrigins.push_back(pKFini);

        mState = OK;

        initID = pKFcur->mnId;
    }

    void Tracking::CreateMapInAtlas()
    {
        mnLastInitFrameId = mCurrentFrame.mnId;
        mpAtlas->CreateNewMap();
        if (mSensor == System::IMU_STEREO || mSensor == System::IMU_MONOCULAR || mSensor == System::IMU_RGBD)
            mpAtlas->SetInertialSensor();
        mbSetInit = false;

        mnInitialFrameId = mCurrentFrame.mnId + 1;
        mState = NO_IMAGES_YET;

        // Restart the variable with information about the last KF
        mbVelocity = false;
        // mnLastRelocFrameId = mnLastInitFrameId; // The last relocation KF_id is the current id, because it is the new starting point for new map
        Verbose::PrintMess("First frame id in map: " + to_string(mnLastInitFrameId + 1), Verbose::VERBOSITY_NORMAL);
        mbVO = false; // Init value for know if there are enough MapPoints in the last KF
        if (mSensor == System::MONOCULAR || mSensor == System::IMU_MONOCULAR)
        {
            mbReadyToInitializate = false;
        }

        if ((mSensor == System::IMU_MONOCULAR || mSensor == System::IMU_STEREO || mSensor == System::IMU_RGBD) && mpImuPreintegratedFromLastKF)
        {
            delete mpImuPreintegratedFromLastKF;
            mpImuPreintegratedFromLastKF = new IMU::Preintegrated(IMU::Bias(), *mpImuCalib);
        }

        if (mpLastKeyFrame)
            mpLastKeyFrame = static_cast<KeyFrame *>(NULL);

        if (mpReferenceKF)
            mpReferenceKF = static_cast<KeyFrame *>(NULL);

        mLastFrame = Frame();
        mCurrentFrame = Frame();
        mvIniMatches.clear();

        mbCreatedMap = true;
    }

    void Tracking::CheckReplacedInLastFrame()
    {
        for (int i = 0; i < mLastFrame.N; i++)
        {
            MapPoint *pMP = mLastFrame.mvpMapPoints[i];

            if (pMP)
            {
                MapPoint *pRep = pMP->GetReplaced();
                if (pRep)
                {
                    mLastFrame.mvpMapPoints[i] = pRep;
                }
            }
        }
    }

    bool Tracking::TrackReferenceKeyFrame()
    {
        const bool ydm_pipe_profile = (std::getenv("YDM_PIPELINE_PROFILE") && std::string(std::getenv("YDM_PIPELINE_PROFILE")) == "1");
        int ydm_pipe_every = 1;
        if (const char* env_pe = std::getenv("YDM_PIPELINE_PROFILE_EVERY"))
        {
            try { ydm_pipe_every = std::stoi(std::string(env_pe)); } catch (...) { ydm_pipe_every = 1; }
            if (ydm_pipe_every < 1) ydm_pipe_every = 1;
        }
        std::chrono::steady_clock::time_point t0;
        std::chrono::steady_clock::time_point t;
        auto ms_since = [](const std::chrono::steady_clock::time_point &a,
                           const std::chrono::steady_clock::time_point &b) -> double {
            return std::chrono::duration_cast<std::chrono::duration<double, std::milli>>(b - a).count();
        };
        double ms_bow = 0.0, ms_match = 0.0, ms_opt = 0.0, ms_out = 0.0;
        if (ydm_pipe_profile) { t0 = t = std::chrono::steady_clock::now(); }

        // Compute Bag of Words vector
        mCurrentFrame.ComputeBoW();

        if (ydm_pipe_profile)
        {
            const auto t1 = std::chrono::steady_clock::now();
            ms_bow = ms_since(t, t1);
            t = t1;
        }

        // We perform first an ORB matching with the reference keyframe
        // If enough matches are found we setup a PnP solver
        ORBmatcher matcher(0.7, true);
        vector<MapPoint *> vpMapPointMatches;

        int nmatches = matcher.SearchByBoW(mpReferenceKF, mCurrentFrame, vpMapPointMatches);

        if (ydm_pipe_profile)
        {
            const auto t1 = std::chrono::steady_clock::now();
            ms_match = ms_since(t, t1);
            t = t1;
        }

        if (nmatches < 15)
        {
            cout << "TRACK_REF_KF: Less than 15 matches!!\n";
            return false;
        }

        mCurrentFrame.mvpMapPoints = vpMapPointMatches;
        mCurrentFrame.SetPose(mLastFrame.GetPose());

        // mCurrentFrame.PrintPointDistribution();

        // cout << " TrackReferenceKeyFrame mLastFrame.mTcw:  " << mLastFrame.mTcw << endl;
        Optimizer::PoseOptimization(&mCurrentFrame);

        if (ydm_pipe_profile)
        {
            const auto t1 = std::chrono::steady_clock::now();
            ms_opt = ms_since(t, t1);
            t = t1;
        }

        // Discard outliers
        int nmatchesMap = 0;
        for (int i = 0; i < mCurrentFrame.N; i++)
        {
            // if(i >= mCurrentFrame.Nleft) break;
            if (mCurrentFrame.mvpMapPoints[i])
            {
                if (mCurrentFrame.mvbOutlier[i])
                {
                    MapPoint *pMP = mCurrentFrame.mvpMapPoints[i];

                    mCurrentFrame.mvpMapPoints[i] = static_cast<MapPoint *>(NULL);
                    mCurrentFrame.mvbOutlier[i] = false;
                    if (i < mCurrentFrame.Nleft)
                    {
                        pMP->mbTrackInView = false;
                    }
                    else
                    {
                        pMP->mbTrackInViewR = false;
                    }
                    pMP->mbTrackInView = false;
                    pMP->mnLastFrameSeen = mCurrentFrame.mnId;
                    nmatches--;
                }
                else if (mCurrentFrame.mvpMapPoints[i]->Observations() > 0)
                    nmatchesMap++;
            }
        }

        if (ydm_pipe_profile)
        {
            const auto t1 = std::chrono::steady_clock::now();
            ms_out = ms_since(t, t1);
            const double ms_total = ms_since(t0, t1);
            if ((mCurrentFrame.mnId % ydm_pipe_every) == 0)
            {
                std::cout << "[YDM][PIPE][REFKF] F=" << std::setw(6) << mCurrentFrame.mnId
                          << " BoW=" << std::setw(6) << std::fixed << std::setprecision(2) << ms_bow << "ms"
                          << " Match=" << std::setw(6) << ms_match << "ms"
                          << " Opt=" << std::setw(6) << ms_opt << "ms"
                          << " Out=" << std::setw(6) << ms_out << "ms"
                          << " || TOTAL=" << std::setw(7) << ms_total << "ms"
                          << std::endl;
            }
        }

        if (mSensor == System::IMU_MONOCULAR || mSensor == System::IMU_STEREO || mSensor == System::IMU_RGBD)
            return true;
        else
            return nmatchesMap >= 10;
    }

    void Tracking::UpdateLastFrame()
    {
        // Update pose according to reference keyframe
        KeyFrame *pRef = mLastFrame.mpReferenceKF;
        Sophus::SE3f Tlr = mlRelativeFramePoses.back();
        mLastFrame.SetPose(Tlr * pRef->GetPose());

        if (mnLastKeyFrameId == mLastFrame.mnId || mSensor == System::MONOCULAR || mSensor == System::IMU_MONOCULAR || !mbOnlyTracking)
            return;

        // Create "visual odometry" MapPoints
        // We sort points according to their measured depth by the stereo/RGB-D sensor
        vector<pair<float, int>> vDepthIdx;
        const int Nfeat = mLastFrame.Nleft == -1 ? mLastFrame.N : mLastFrame.Nleft;
        vDepthIdx.reserve(Nfeat);
        for (int i = 0; i < Nfeat; i++)
        {
            float z = mLastFrame.mvDepth[i];
            if (z > 0)
            {
                vDepthIdx.push_back(make_pair(z, i));
            }
        }

        if (vDepthIdx.empty())
            return;

        sort(vDepthIdx.begin(), vDepthIdx.end());

        // We insert all close points (depth<mThDepth)
        // If less than 100 close points, we insert the 100 closest ones.
        int nPoints = 0;
        for (size_t j = 0; j < vDepthIdx.size(); j++)
        {
            int i = vDepthIdx[j].second;

            bool bCreateNew = false;

            MapPoint *pMP = mLastFrame.mvpMapPoints[i];

            if (!pMP)
                bCreateNew = true;
            else if (pMP->Observations() < 1)
                bCreateNew = true;

            if (bCreateNew)
            {
                Eigen::Vector3f x3D;

                if (mLastFrame.Nleft == -1)
                {
                    mLastFrame.UnprojectStereo(i, x3D);
                }
                else
                {
                    x3D = mLastFrame.UnprojectStereoFishEye(i);
                }

                MapPoint *pNewMP = new MapPoint(x3D, mpAtlas->GetCurrentMap(), &mLastFrame, i);
                mLastFrame.mvpMapPoints[i] = pNewMP;

                mlpTemporalPoints.push_back(pNewMP);
                nPoints++;
            }
            else
            {
                nPoints++;
            }

            if (vDepthIdx[j].first > mThDepth && nPoints > 100)
                break;
        }
    }

    bool Tracking::TrackWithMotionModel()
    {
        const bool ydm_pipe_profile = (std::getenv("YDM_PIPELINE_PROFILE") && std::string(std::getenv("YDM_PIPELINE_PROFILE")) == "1");
        int ydm_pipe_every = 1;
        if (const char* env_pe = std::getenv("YDM_PIPELINE_PROFILE_EVERY"))
        {
            try { ydm_pipe_every = std::stoi(std::string(env_pe)); } catch (...) { ydm_pipe_every = 1; }
            if (ydm_pipe_every < 1) ydm_pipe_every = 1;
        }
        std::chrono::steady_clock::time_point t0;
        std::chrono::steady_clock::time_point t;
        auto ms_since = [](const std::chrono::steady_clock::time_point &a,
                           const std::chrono::steady_clock::time_point &b) -> double {
            return std::chrono::duration_cast<std::chrono::duration<double, std::milli>>(b - a).count();
        };
        double ms_update_last = 0.0, ms_pose_pred = 0.0;
        double ms_residual = 0.0, ms_cleaner = 0.0, ms_verify = 0.0;
        double ms_match_proj = 0.0, ms_match_wide = 0.0, ms_opt = 0.0, ms_out = 0.0;
        if (ydm_pipe_profile) { t0 = t = std::chrono::steady_clock::now(); }

        ORBmatcher matcher(0.9, true);

        // Update last frame pose according to its reference keyframe
        // Create "visual odometry" points if in Localization Mode
        UpdateLastFrame();

        if (ydm_pipe_profile)
        {
            const auto t1 = std::chrono::steady_clock::now();
            ms_update_last = ms_since(t, t1);
            t = t1;
        }

        if (mpAtlas->isImuInitialized() && (mCurrentFrame.mnId > mnLastRelocFrameId + mnFramesToResetIMU))
        {
            // Predict state with IMU if it is initialized and it doesnt need reset
            PredictStateIMU();
            return true;
        }
        else
        {
            mCurrentFrame.SetPose(mVelocity * mLastFrame.GetPose());
        }

        if (ydm_pipe_profile)
        {
            const auto t1 = std::chrono::steady_clock::now();
            ms_pose_pred = ms_since(t, t1);
            t = t1;
        }

        // DIO-SLAM style dynamic keypoint filtering (requires predicted pose)
        // Compute FlowNet + ego-compensated residual ONCE, then reuse across modules.
        // NEW: If YDM_BG_PEEL=1, use iterative background peeling with GN pose refinement (Option B)
        if ((mSensor == System::RGBD || mSensor == System::IMU_RGBD) &&
            !mLastImGray.empty() && !mPrevDepthMeters.empty())
        {
            const Sophus::SE3f TcwPrev = mLastFrame.GetPose();
            Sophus::SE3f TcwCurr = mCurrentFrame.GetPose();  // Initial pose from motion model

            cv::Mat flowPrevToCurr, residual, validMask, motionMask;
            float thr = 0.0f;
            bool okResidual = false;

            // Check if BG peeling is enabled
            const char* env_bg_peel = std::getenv("YDM_BG_PEEL");
            const bool use_bg_peel = (env_bg_peel && std::string(env_bg_peel) == "1");

            if (use_bg_peel)
            {
                cv::Mat staticPriorMask;
                if (!mPrevDepthMeters.empty() && mPrevDepthMeters.type() == CV_32FC1)
                {
                    cv::Mat validDepthMask = cv::Mat::zeros(mPrevDepthMeters.size(), CV_8UC1);
                    for (int y = 0; y < mPrevDepthMeters.rows; ++y)
                    {
                        const float *dptr = mPrevDepthMeters.ptr<float>(y);
                        uchar *vptr = validDepthMask.ptr<uchar>(y);
                        for (int x = 0; x < mPrevDepthMeters.cols; ++x)
                        {
                            const float z = dptr[x];
                            if (z > 0.0f && std::isfinite(z))
                                vptr[x] = 255;
                        }
                    }

                    cv::Mat dynamicSemanticMask;
                    if (!mCurrInstanceIdMask.empty() &&
                        mCurrInstanceIdMask.type() == CV_32SC1 &&
                        mCurrInstanceIdMask.size() == validDepthMask.size() &&
                        !mCurrInstanceClassNames.empty() &&
                        mpDetector)
                    {
                        const int nInst = static_cast<int>(mCurrInstanceClassNames.size());
                        std::vector<uchar> instDynamic(nInst, 0);
                        const std::vector<std::string> &dynNames = mpDetector->mvDynamicNames;
                        for (int i = 0; i < nInst; ++i)
                        {
                            const std::string &cls = mCurrInstanceClassNames[i];
                            if (std::find(dynNames.begin(), dynNames.end(), cls) != dynNames.end())
                                instDynamic[i] = 1;
                        }

                        dynamicSemanticMask = cv::Mat::zeros(validDepthMask.size(), CV_8UC1);
                        for (int y = 0; y < mCurrInstanceIdMask.rows; ++y)
                        {
                            const int *iptr = mCurrInstanceIdMask.ptr<int>(y);
                            uchar *mptr = dynamicSemanticMask.ptr<uchar>(y);
                            for (int x = 0; x < mCurrInstanceIdMask.cols; ++x)
                            {
                                const int id = iptr[x];
                                if (id > 0 && id <= nInst && instDynamic[id - 1])
                                    mptr[x] = 255;
                            }
                        }
                    }

                    if (!dynamicSemanticMask.empty())
                    {
                        cv::Mat invDynamic;
                        cv::bitwise_not(dynamicSemanticMask, invDynamic);
                        cv::bitwise_and(validDepthMask, invDynamic, staticPriorMask);
                    }
                    else
                    {
                        staticPriorMask = validDepthMask;
                    }
                }

                // Option B: Iterative BG peeling (pose unchanged, only refine mask)
                okResidual = mDynamicObjectCleaner.RefineMotionWithFlowPeeling(
                    mLastImGray,
                    mPrevDepthMeters,
                    TcwPrev,
                    mImGray,
                    mCurrentFrame.mK_,
                    TcwCurr,  // Used for computation but NOT updated
                    flowPrevToCurr,
                    residual,
                    validMask,
                    motionMask,
                    thr,
                    staticPriorMask);
                // Note: pose remains unchanged, we only get refined residual/mask
            }
            else
            {
                // Original: Single-shot residual computation
                okResidual = mDynamicObjectCleaner.ComputeFlowAndResidualRGBD(
                    mLastImGray,
                    mPrevDepthMeters,
                    TcwPrev,
                    mImGray,
                    TcwCurr,
                    mCurrentFrame.mK_,
                    flowPrevToCurr,
                    residual,
                    validMask,
                    motionMask,
                    thr);
            }

            if (ydm_pipe_profile)
            {
                const auto t1 = std::chrono::steady_clock::now();
                ms_residual = ms_since(t, t1);
                t = t1;
            }

            if (okResidual)
            {
                mCurrFlowPrevToCurr = flowPrevToCurr;
                mCurrResidual = residual;
                mCurrResidualValidMask = validMask;
                mCurrResidualMotionMask = motionMask;
                mCurrResidualThr = thr;
                mCurrResidualFrameId = mCurrentFrame.mnId;

                // If we have instance segmentation, produce refined semantic+motion mask and outliers.
                if (!mCurrInstanceIdMask.empty() && mCurrInstanceIdMask.type() == CV_32SC1)
                {
                    std::vector<bool> outFlags;
                    cv::Mat refinedMask;
                    cv::Mat dynamicRigidMask;
                    const bool ok = mDynamicObjectCleaner.CleanRGBDFromPrecomputed(
                        mCurrResidual,
                        mCurrResidualValidMask,
                        mCurrResidualMotionMask,
                        mCurrResidualThr,
                        mCurrInstanceIdMask,
                        mCurrInstanceClassNames,
                        mPrevDynamicRigidMask,
                        mCurrFlowPrevToCurr,
                        mCurrentFrame.mvKeysUn,
                        outFlags,
                        refinedMask,
                        dynamicRigidMask);

                    if (ydm_pipe_profile)
                    {
                        const auto t1 = std::chrono::steady_clock::now();
                        ms_cleaner = ms_since(t, t1);
                        t = t1;
                    }

                    if (ok && !outFlags.empty())
                    {
                        // Mark outliers early so matching utilities will skip them
                        const size_t n = std::min(outFlags.size(), mCurrentFrame.mvbOutlier.size());
                        for (size_t i = 0; i < n; ++i)
                            if (outFlags[i])
                                mCurrentFrame.mvbOutlier[i] = true;

                        mCurrentFrame.mvDynamicMask.clear();
                        mCurrentFrame.mvDynamicMask.push_back(refinedMask);

                        mCurrDynamicRigidMask = dynamicRigidMask.clone();
                    }
                }

                // Apply additional depth+box logic using the same residual cache (no extra FlowNet).
                VerifyAndPruneFeatures();

                if (ydm_pipe_profile)
                {
                    const auto t1 = std::chrono::steady_clock::now();
                    ms_verify = ms_since(t, t1);
                    t = t1;
                }
            }
        }

        fill(mCurrentFrame.mvpMapPoints.begin(), mCurrentFrame.mvpMapPoints.end(), static_cast<MapPoint *>(NULL));

        // Project points seen in previous frame
        int th;

        if (mSensor == System::STEREO)
            th = 7;
        else
            th = 15;

        int nmatches = matcher.SearchByProjection(mCurrentFrame, mLastFrame, th, mSensor == System::MONOCULAR || mSensor == System::IMU_MONOCULAR);

        if (ydm_pipe_profile)
        {
            const auto t1 = std::chrono::steady_clock::now();
            ms_match_proj = ms_since(t, t1);
            t = t1;
        }

        // If few matches, uses a wider window search
        if (nmatches < 20)
        {
            Verbose::PrintMess("Not enough matches, wider window search!!", Verbose::VERBOSITY_NORMAL);
            fill(mCurrentFrame.mvpMapPoints.begin(), mCurrentFrame.mvpMapPoints.end(), static_cast<MapPoint *>(NULL));

            nmatches = matcher.SearchByProjection(mCurrentFrame, mLastFrame, 2 * th, mSensor == System::MONOCULAR || mSensor == System::IMU_MONOCULAR);
            Verbose::PrintMess("Matches with wider search: " + to_string(nmatches), Verbose::VERBOSITY_NORMAL);

            if (ydm_pipe_profile)
            {
                const auto t1 = std::chrono::steady_clock::now();
                ms_match_wide = ms_since(t, t1);
                t = t1;
            }
        }

        if (nmatches < 20)
        {
            Verbose::PrintMess("Not enough matches!!", Verbose::VERBOSITY_NORMAL);
            if (mSensor == System::IMU_MONOCULAR || mSensor == System::IMU_STEREO || mSensor == System::IMU_RGBD)
                return true;
            else
                return false;
        }

        // Optimize frame pose with all matches
        Optimizer::PoseOptimization(&mCurrentFrame);

        if (ydm_pipe_profile)
        {
            const auto t1 = std::chrono::steady_clock::now();
            ms_opt = ms_since(t, t1);
            t = t1;
        }

        // Discard outliers
        int nmatchesMap = 0;
        for (int i = 0; i < mCurrentFrame.N; i++)
        {
            if (mCurrentFrame.mvpMapPoints[i])
            {
                if (mCurrentFrame.mvbOutlier[i])
                {
                    MapPoint *pMP = mCurrentFrame.mvpMapPoints[i];

                    mCurrentFrame.mvpMapPoints[i] = static_cast<MapPoint *>(NULL);
                    mCurrentFrame.mvbOutlier[i] = false;
                    if (i < mCurrentFrame.Nleft)
                    {
                        pMP->mbTrackInView = false;
                    }
                    else
                    {
                        pMP->mbTrackInViewR = false;
                    }
                    pMP->mnLastFrameSeen = mCurrentFrame.mnId;
                    nmatches--;
                }
                else if (mCurrentFrame.mvpMapPoints[i]->Observations() > 0)
                    nmatchesMap++;
            }
        }

        if (ydm_pipe_profile)
        {
            const auto t1 = std::chrono::steady_clock::now();
            ms_out = ms_since(t, t1);
            const double ms_total = ms_since(t0, t1);
            if ((mCurrentFrame.mnId % ydm_pipe_every) == 0)
            {
                std::cout << "[YDM][PIPE][MM] F=" << std::setw(6) << mCurrentFrame.mnId
                          << " UpdLast=" << std::setw(6) << std::fixed << std::setprecision(2) << ms_update_last << "ms"
                          << " PosePred=" << std::setw(6) << ms_pose_pred << "ms"
                          << " Residual=" << std::setw(6) << ms_residual << "ms"
                          << " Cleaner=" << std::setw(6) << ms_cleaner << "ms"
                          << " Verify=" << std::setw(6) << ms_verify << "ms"
                          << " Match=" << std::setw(6) << ms_match_proj << "ms"
                          << " Wide=" << std::setw(6) << ms_match_wide << "ms"
                          << " Opt=" << std::setw(6) << ms_opt << "ms"
                          << " Out=" << std::setw(6) << ms_out << "ms"
                          << " || TOTAL=" << std::setw(7) << ms_total << "ms"
                          << std::endl;
            }
        }

        if (mbOnlyTracking)
        {
            mbVO = nmatchesMap < 10;
            return nmatches > 20;
        }

        if (mSensor == System::IMU_MONOCULAR || mSensor == System::IMU_STEREO || mSensor == System::IMU_RGBD)
            return true;
        else
            return nmatchesMap >= 10;
    }

    bool Tracking::TrackLocalMap()
    {

        // We have an estimation of the camera pose and some map points tracked in the frame.
        // We retrieve the local map and try to find matches to points in the local map.
        mTrackedFr++;

        UpdateLocalMap();
        SearchLocalPoints();

        // TOO check outliers before PO
        int aux1 = 0, aux2 = 0;
        for (int i = 0; i < mCurrentFrame.N; i++)
            if (mCurrentFrame.mvpMapPoints[i])
            {
                aux1++;
                if (mCurrentFrame.mvbOutlier[i])
                    aux2++;
            }

        int inliers;
        if (!mpAtlas->isImuInitialized())
            Optimizer::PoseOptimization(&mCurrentFrame);
        else
        {
            if (mCurrentFrame.mnId <= mnLastRelocFrameId + mnFramesToResetIMU)
            {
                Verbose::PrintMess("TLM: PoseOptimization ", Verbose::VERBOSITY_DEBUG);
                Optimizer::PoseOptimization(&mCurrentFrame);
            }
            else
            {
                // if(!mbMapUpdated && mState == OK) //  && (mnMatchesInliers>30))
                if (!mbMapUpdated) //  && (mnMatchesInliers>30))
                {
                    Verbose::PrintMess("TLM: PoseInertialOptimizationLastFrame ", Verbose::VERBOSITY_DEBUG);
                    inliers = Optimizer::PoseInertialOptimizationLastFrame(&mCurrentFrame); // , !mpLastKeyFrame->GetMap()->GetIniertialBA1());
                }
                else
                {
                    Verbose::PrintMess("TLM: PoseInertialOptimizationLastKeyFrame ", Verbose::VERBOSITY_DEBUG);
                    inliers = Optimizer::PoseInertialOptimizationLastKeyFrame(&mCurrentFrame); // , !mpLastKeyFrame->GetMap()->GetIniertialBA1());
                }
            }
        }

        aux1 = 0, aux2 = 0;
        for (int i = 0; i < mCurrentFrame.N; i++)
            if (mCurrentFrame.mvpMapPoints[i])
            {
                aux1++;
                if (mCurrentFrame.mvbOutlier[i])
                    aux2++;
            }

        mnMatchesInliers = 0;

        // Update MapPoints Statistics
        for (int i = 0; i < mCurrentFrame.N; i++)
        {
            if (mCurrentFrame.mvpMapPoints[i])
            {
                if (!mCurrentFrame.mvbOutlier[i])
                {
                    mCurrentFrame.mvpMapPoints[i]->IncreaseFound();
                    if (!mbOnlyTracking)
                    {
                        if (mCurrentFrame.mvpMapPoints[i]->Observations() > 0)
                            mnMatchesInliers++;
                    }
                    else
                        mnMatchesInliers++;
                }
                else if (mSensor == System::STEREO)
                    mCurrentFrame.mvpMapPoints[i] = static_cast<MapPoint *>(NULL);
            }
        }

        // Decide if the tracking was succesful
        // More restrictive if there was a relocalization recently
        mpLocalMapper->mnMatchesInliers = mnMatchesInliers;
        if (mCurrentFrame.mnId < mnLastRelocFrameId + mMaxFrames && mnMatchesInliers < 50)
            return false;

        if ((mnMatchesInliers > 10) && (mState == RECENTLY_LOST))
            return true;

        if (mSensor == System::IMU_MONOCULAR)
        {
            if ((mnMatchesInliers < 15 && mpAtlas->isImuInitialized()) || (mnMatchesInliers < 50 && !mpAtlas->isImuInitialized()))
            {
                return false;
            }
            else
                return true;
        }
        else if (mSensor == System::IMU_STEREO || mSensor == System::IMU_RGBD)
        {
            if (mnMatchesInliers < 15)
            {
                return false;
            }
            else
                return true;
        }
        else
        {
            if (mnMatchesInliers < 30)
                return false;
            else
                return true;
        }
    }

    bool Tracking::NeedNewKeyFrame()
    {
        const bool ydm_kf_debug = (std::getenv("YDM_KF_DEBUG") &&
                                   std::string(std::getenv("YDM_KF_DEBUG")) == "1");
        auto kf_log = [&](const std::string &msg)
        {
            if (ydm_kf_debug)
                std::cout << "[YDM][KF] F=" << mCurrentFrame.mnId << " " << msg << std::endl;
        };

        if ((mSensor == System::IMU_MONOCULAR || mSensor == System::IMU_STEREO || mSensor == System::IMU_RGBD) && !mpAtlas->GetCurrentMap()->isImuInitialized())
        {
            if (mSensor == System::IMU_MONOCULAR && (mCurrentFrame.mTimeStamp - mpLastKeyFrame->mTimeStamp) >= 0.25)
                return true;
            else if ((mSensor == System::IMU_STEREO || mSensor == System::IMU_RGBD) && (mCurrentFrame.mTimeStamp - mpLastKeyFrame->mTimeStamp) >= 0.25)
                return true;
            else
            {
                kf_log("IMU not initialized -> false");
                return false;
            }
        }

        if (mbOnlyTracking)
        {
            kf_log("mbOnlyTracking=1 -> false");
            return false;
        }

        // If Local Mapping is freezed by a Loop Closure do not insert keyframes
        if (mpLocalMapper->isStopped() || mpLocalMapper->stopRequested())
        {
            /*if(mSensor == System::MONOCULAR)
            {
                std::cout << "NeedNewKeyFrame: localmap stopped" << std::endl;
            }*/
            kf_log("LocalMapping stopped -> false");
            return false;
        }

        const int nKFs = mpAtlas->KeyFramesInMap();

        // Do not insert keyframes if not enough frames have passed from last relocalisation
        if (mCurrentFrame.mnId < mnLastRelocFrameId + mMaxFrames && nKFs > mMaxFrames)
        {
            kf_log("Reloc window -> false");
            return false;
        }

        // Tracked MapPoints in the reference keyframe
        int nMinObs = 3;
        if (nKFs <= 2)
            nMinObs = 2;
        int nRefMatches = mpReferenceKF->TrackedMapPoints(nMinObs);

        // Local Mapping accept keyframes?
        bool bLocalMappingIdle = mpLocalMapper->AcceptKeyFrames();

        // Check how many "close" points are being tracked and how many could be potentially created.
        int nNonTrackedClose = 0;
        int nTrackedClose = 0;

        if (mSensor != System::MONOCULAR && mSensor != System::IMU_MONOCULAR)
        {
            int N = (mCurrentFrame.Nleft == -1) ? mCurrentFrame.N : mCurrentFrame.Nleft;
            for (int i = 0; i < N; i++)
            {
                if (mCurrentFrame.mvDepth[i] > 0 && mCurrentFrame.mvDepth[i] < mThDepth)
                {
                    if (mCurrentFrame.mvpMapPoints[i] && !mCurrentFrame.mvbOutlier[i])
                        nTrackedClose++;
                    else
                        nNonTrackedClose++;
                }
            }
            // Verbose::PrintMess("[NEEDNEWKF]-> closed points: " + to_string(nTrackedClose) + "; non tracked closed points: " + to_string(nNonTrackedClose), Verbose::VERBOSITY_NORMAL);// Verbose::VERBOSITY_DEBUG);
        }

        bool bNeedToInsertClose;
        bNeedToInsertClose = (nTrackedClose < 100) && (nNonTrackedClose > 70);

        // Thresholds
        float thRefRatio = 0.75f;
        if (nKFs < 2)
            thRefRatio = 0.4f;

        /*int nClosedPoints = nTrackedClose + nNonTrackedClose;
        const int thStereoClosedPoints = 15;
        if(nClosedPoints < thStereoClosedPoints && (mSensor==System::STEREO || mSensor==System::IMU_STEREO))
        {
            //Pseudo-monocular, there are not enough close points to be confident about the stereo observations.
            thRefRatio = 0.9f;
        }*/

        if (mSensor == System::MONOCULAR)
            thRefRatio = 0.9f;

        if (mpCamera2)
            thRefRatio = 0.75f;

        if (mSensor == System::IMU_MONOCULAR)
        {
            if (mnMatchesInliers > 350) // Points tracked from the local map
                thRefRatio = 0.75f;
            else
                thRefRatio = 0.90f;
        }

        // Condition 1a: More than "MaxFrames" have passed from last keyframe insertion
        const bool c1a = mCurrentFrame.mnId >= mnLastKeyFrameId + mMaxFrames;
        // Condition 1b: More than "MinFrames" have passed and Local Mapping is idle
        const bool c1b = ((mCurrentFrame.mnId >= mnLastKeyFrameId + mMinFrames) && bLocalMappingIdle); // mpLocalMapper->KeyframesInQueue() < 2);
        // Condition 1c: tracking is weak
        const bool c1c = mSensor != System::MONOCULAR && mSensor != System::IMU_MONOCULAR && mSensor != System::IMU_STEREO && mSensor != System::IMU_RGBD && (mnMatchesInliers < nRefMatches * 0.25 || bNeedToInsertClose);
        // Condition 2: Few tracked points compared to reference keyframe. Lots of visual odometry compared to map matches.
        const bool c2 = (((mnMatchesInliers < nRefMatches * thRefRatio || bNeedToInsertClose)) && mnMatchesInliers > 15);

        // std::cout << "NeedNewKF: c1a=" << c1a << "; c1b=" << c1b << "; c1c=" << c1c << "; c2=" << c2 << std::endl;
        //  Temporal condition for Inertial cases
        bool c3 = false;
        if (mpLastKeyFrame)
        {
            if (mSensor == System::IMU_MONOCULAR)
            {
                if ((mCurrentFrame.mTimeStamp - mpLastKeyFrame->mTimeStamp) >= 0.5)
                    c3 = true;
            }
            else if (mSensor == System::IMU_STEREO || mSensor == System::IMU_RGBD)
            {
                if ((mCurrentFrame.mTimeStamp - mpLastKeyFrame->mTimeStamp) >= 0.5)
                    c3 = true;
            }
        }

        bool c4 = false;
        if ((((mnMatchesInliers < 75) && (mnMatchesInliers > 15)) || mState == RECENTLY_LOST) && (mSensor == System::IMU_MONOCULAR)) // MODIFICATION_2, originally ((((mnMatchesInliers<75) && (mnMatchesInliers>15)) || mState==RECENTLY_LOST) && ((mSensor == System::IMU_MONOCULAR)))
            c4 = true;
        else
            c4 = false;

        if (ydm_kf_debug)
        {
            std::ostringstream oss;
            oss << "nKFs=" << nKFs
                << " nRefMatches=" << nRefMatches
                << " inliers=" << mnMatchesInliers
                << " nTrackedClose=" << nTrackedClose
                << " nNonTrackedClose=" << nNonTrackedClose
                << " bNeedToInsertClose=" << (bNeedToInsertClose ? 1 : 0)
                << " c1a=" << c1a
                << " c1b=" << c1b
                << " c1c=" << c1c
                << " c2=" << c2
                << " c3=" << c3
                << " c4=" << c4
                << " LocalMappingIdle=" << (bLocalMappingIdle ? 1 : 0);
            kf_log(oss.str());
        }

        if (((c1a || c1b || c1c) && c2) || c3 || c4)
        {
            // If the mapping accepts keyframes, insert keyframe.
            // Otherwise send a signal to interrupt BA
            if (bLocalMappingIdle || mpLocalMapper->IsInitializing())
            {
                kf_log("Insert KF -> true");
                return true;
            }
            else
            {
                mpLocalMapper->InterruptBA();
                if (mSensor != System::MONOCULAR && mSensor != System::IMU_MONOCULAR)
                {
                    if (mpLocalMapper->KeyframesInQueue() < 3)
                    {
                        kf_log("InterruptBA + queue<3 -> true");
                        return true;
                    }
                    else
                    {
                        kf_log("InterruptBA + queue>=3 -> false");
                        return false;
                    }
                }
                else
                {
                    // std::cout << "NeedNewKeyFrame: localmap is busy" << std::endl;
                    kf_log("LocalMapping busy (mono) -> false");
                    return false;
                }
            }
        }
        else
        {
            kf_log("Conditions not met -> false");
            return false;
        }
    }

    void Tracking::CreateNewKeyFrame()
    {
        if (mpLocalMapper->IsInitializing() && !mpAtlas->isImuInitialized())
            return;

        if (!mpLocalMapper->SetNotStop(true))
            return;

        KeyFrame *pKF = new KeyFrame(mCurrentFrame, mpAtlas->GetCurrentMap(), mpKeyFrameDB);

        if (mpAtlas->isImuInitialized()) //  || mpLocalMapper->IsInitializing())
            pKF->bImu = true;

        pKF->SetNewBias(mCurrentFrame.mImuBias);
        mpReferenceKF = pKF;
        mCurrentFrame.mpReferenceKF = pKF;

        if (mpLastKeyFrame)
        {
            pKF->mPrevKF = mpLastKeyFrame;
            mpLastKeyFrame->mNextKF = pKF;
        }
        else
            Verbose::PrintMess("No last KF in KF creation!!", Verbose::VERBOSITY_NORMAL);

        // Reset preintegration from last KF (Create new object)
        if (mSensor == System::IMU_MONOCULAR || mSensor == System::IMU_STEREO || mSensor == System::IMU_RGBD)
        {
            mpImuPreintegratedFromLastKF = new IMU::Preintegrated(pKF->GetImuBias(), pKF->mImuCalib);
        }


        
        if (mSensor != System::MONOCULAR && mSensor != System::IMU_MONOCULAR) // TODO check if incluide imu_stereo
        {
            mCurrentFrame.UpdatePoseMatrices();
            // cout << "create new MPs" << endl;
            // We sort points by the measured depth by the stereo/RGBD sensor.
            // We create all those MapPoints whose depth < mThDepth.
            // If there are less than 100 close points we create the 100 closest.
            int maxPoint = 100;
            if (mSensor == System::IMU_STEREO || mSensor == System::IMU_RGBD)
                maxPoint = 100;

            vector<pair<float, int>> vDepthIdx;
            int N = (mCurrentFrame.Nleft != -1) ? mCurrentFrame.Nleft : mCurrentFrame.N;
            vDepthIdx.reserve(mCurrentFrame.N);
            for (int i = 0; i < N; i++)
            {
                float z = mCurrentFrame.mvDepth[i];
                if (z > 0)
                {
                    vDepthIdx.push_back(make_pair(z, i));
                }
            }

            if (!vDepthIdx.empty())
            {
                sort(vDepthIdx.begin(), vDepthIdx.end());

                int nPoints = 0;
                for (size_t j = 0; j < vDepthIdx.size(); j++)
                {
                    int i = vDepthIdx[j].second;

                    bool bCreateNew = false;

                    MapPoint *pMP = mCurrentFrame.mvpMapPoints[i];
                    if (!pMP)
                        bCreateNew = true;
                    else if (pMP->Observations() < 1)
                    {
                        bCreateNew = true;
                        mCurrentFrame.mvpMapPoints[i] = static_cast<MapPoint *>(NULL);
                    }

                    if (bCreateNew)
                    {
                        Eigen::Vector3f x3D;

                        if (mCurrentFrame.Nleft == -1)
                        {
                            mCurrentFrame.UnprojectStereo(i, x3D);
                        }
                        else
                        {
                            x3D = mCurrentFrame.UnprojectStereoFishEye(i);
                        }

                        MapPoint *pNewMP = new MapPoint(x3D, pKF, mpAtlas->GetCurrentMap());
                        pNewMP->AddObservation(pKF, i);

                        // Check if it is a stereo observation in order to not
                        // duplicate mappoints
                        if (mCurrentFrame.Nleft != -1 && mCurrentFrame.mvLeftToRightMatch[i] >= 0)
                        {
                            mCurrentFrame.mvpMapPoints[mCurrentFrame.Nleft + mCurrentFrame.mvLeftToRightMatch[i]] = pNewMP;
                            pNewMP->AddObservation(pKF, mCurrentFrame.Nleft + mCurrentFrame.mvLeftToRightMatch[i]);
                            pKF->AddMapPoint(pNewMP, mCurrentFrame.Nleft + mCurrentFrame.mvLeftToRightMatch[i]);
                        }

                        pKF->AddMapPoint(pNewMP, i);
                        pNewMP->ComputeDistinctiveDescriptors();
                        pNewMP->UpdateNormalAndDepth();
                        mpAtlas->AddMapPoint(pNewMP);

                        mCurrentFrame.mvpMapPoints[i] = pNewMP;
                        nPoints++;
                    }
                    else
                    {
                        nPoints++;
                    }

                    if (vDepthIdx[j].first > mThDepth && nPoints > maxPoint)
                    {
                        break;
                    }
                }
                // Verbose::PrintMess("new mps for stereo KF: " + to_string(nPoints), Verbose::VERBOSITY_NORMAL);
            }
        }

        mpLocalMapper->InsertKeyFrame(pKF);

        // 保存RGB和深度图用于点云生成
        if(mSensor == System::RGBD || mSensor == System::IMU_RGBD)
        {
            pKF->imLeftRgb = mImRGB.clone();
            pKF->imDepth = mImDepth.clone();
            // 使用 mCurrentFrame 的 mask（refined后的最终版本），并做深拷贝
            // 说明：cv::Mat 默认是浅拷贝（共享底层数据）。这里逐张 clone()，避免后续对 Frame/Viewer
            // 的 in-place 操作或覆盖导致 KeyFrame 中的 mask 被意外影响。
            pKF->mvDynamicMask.clear();
            pKF->mvDynamicMask.reserve(mCurrentFrame.mvDynamicMask.size());
            for (const cv::Mat &m : mCurrentFrame.mvDynamicMask)
            {
                pKF->mvDynamicMask.push_back(m.empty() ? cv::Mat() : m.clone());
            }
            pKF->mvDynamicArea = mCurrentFrame.mvDynamicArea;
        }
        
        // 将关键帧传递给点云建图模块（仅用于RGB-D）
        if(mpPointCloudMapping && (mSensor == System::RGBD || mSensor == System::IMU_RGBD))
        {
            mpPointCloudMapping->insertKeyFrame(pKF);
        }

        mpLocalMapper->SetNotStop(false);

        mnLastKeyFrameId = mCurrentFrame.mnId;
        mpLastKeyFrame = pKF;
    }

    void Tracking::SearchLocalPoints()
    {
        // Do not search map points already matched
        for (vector<MapPoint *>::iterator vit = mCurrentFrame.mvpMapPoints.begin(), vend = mCurrentFrame.mvpMapPoints.end(); vit != vend; vit++)
        {
            MapPoint *pMP = *vit;
            if (pMP)
            {
                if (pMP->isBad())
                {
                    *vit = static_cast<MapPoint *>(NULL);
                }
                else
                {
                    pMP->IncreaseVisible();
                    pMP->mnLastFrameSeen = mCurrentFrame.mnId;
                    pMP->mbTrackInView = false;
                    pMP->mbTrackInViewR = false;
                }
            }
        }

        int nToMatch = 0;

        // Project points in frame and check its visibility
        for (vector<MapPoint *>::iterator vit = mvpLocalMapPoints.begin(), vend = mvpLocalMapPoints.end(); vit != vend; vit++)
        {
            MapPoint *pMP = *vit;

            if (pMP->mnLastFrameSeen == mCurrentFrame.mnId)
                continue;
            if (pMP->isBad())
                continue;
            // Project (this fills MapPoint variables for matching)
            if (mCurrentFrame.isInFrustum(pMP, 0.5))
            {
                pMP->IncreaseVisible();
                nToMatch++;
            }
            if (pMP->mbTrackInView)
            {
                mCurrentFrame.mmProjectPoints[pMP->mnId] = cv::Point2f(pMP->mTrackProjX, pMP->mTrackProjY);
            }
        }

        if (nToMatch > 0)
        {
            ORBmatcher matcher(0.8);
            int th = 1;
            if (mSensor == System::RGBD || mSensor == System::IMU_RGBD)
                th = 3;
            if (mpAtlas->isImuInitialized())
            {
                if (mpAtlas->GetCurrentMap()->GetIniertialBA2())
                    th = 2;
                else
                    th = 6;
            }
            else if (!mpAtlas->isImuInitialized() && (mSensor == System::IMU_MONOCULAR || mSensor == System::IMU_STEREO || mSensor == System::IMU_RGBD))
            {
                th = 10;
            }

            // If the camera has been relocalised recently, perform a coarser search
            if (mCurrentFrame.mnId < mnLastRelocFrameId + 2)
                th = 5;

            if (mState == LOST || mState == RECENTLY_LOST) // Lost for less than 1 second
                th = 15;                                   // 15

            int matches = matcher.SearchByProjection(mCurrentFrame, mvpLocalMapPoints, th, mpLocalMapper->mbFarPoints, mpLocalMapper->mThFarPoints);
        }
    }

    void Tracking::UpdateLocalMap()
    {
        // This is for visualization
        mpAtlas->SetReferenceMapPoints(mvpLocalMapPoints);

        // Update
        UpdateLocalKeyFrames();
        UpdateLocalPoints();
    }

    void Tracking::UpdateLocalPoints()
    {
        mvpLocalMapPoints.clear();

        int count_pts = 0;

        for (vector<KeyFrame *>::const_reverse_iterator itKF = mvpLocalKeyFrames.rbegin(), itEndKF = mvpLocalKeyFrames.rend(); itKF != itEndKF; ++itKF)
        {
            KeyFrame *pKF = *itKF;
            const vector<MapPoint *> vpMPs = pKF->GetMapPointMatches();

            for (vector<MapPoint *>::const_iterator itMP = vpMPs.begin(), itEndMP = vpMPs.end(); itMP != itEndMP; itMP++)
            {

                MapPoint *pMP = *itMP;
                if (!pMP)
                    continue;
                if (pMP->mnTrackReferenceForFrame == mCurrentFrame.mnId)
                    continue;
                if (!pMP->isBad())
                {
                    count_pts++;
                    mvpLocalMapPoints.push_back(pMP);
                    pMP->mnTrackReferenceForFrame = mCurrentFrame.mnId;
                }
            }
        }
    }

    void Tracking::UpdateLocalKeyFrames()
    {
        // Each map point vote for the keyframes in which it has been observed
        map<KeyFrame *, int> keyframeCounter;
        if (!mpAtlas->isImuInitialized() || (mCurrentFrame.mnId < mnLastRelocFrameId + 2))
        {
            for (int i = 0; i < mCurrentFrame.N; i++)
            {
                MapPoint *pMP = mCurrentFrame.mvpMapPoints[i];
                if (pMP)
                {
                    if (!pMP->isBad())
                    {
                        const map<KeyFrame *, tuple<int, int>> observations = pMP->GetObservations();
                        for (map<KeyFrame *, tuple<int, int>>::const_iterator it = observations.begin(), itend = observations.end(); it != itend; it++)
                            keyframeCounter[it->first]++;
                    }
                    else
                    {
                        mCurrentFrame.mvpMapPoints[i] = NULL;
                    }
                }
            }
        }
        else
        {
            for (int i = 0; i < mLastFrame.N; i++)
            {
                // Using lastframe since current frame has not matches yet
                if (mLastFrame.mvpMapPoints[i])
                {
                    MapPoint *pMP = mLastFrame.mvpMapPoints[i];
                    if (!pMP)
                        continue;
                    if (!pMP->isBad())
                    {
                        const map<KeyFrame *, tuple<int, int>> observations = pMP->GetObservations();
                        for (map<KeyFrame *, tuple<int, int>>::const_iterator it = observations.begin(), itend = observations.end(); it != itend; it++)
                            keyframeCounter[it->first]++;
                    }
                    else
                    {
                        // MODIFICATION
                        mLastFrame.mvpMapPoints[i] = NULL;
                    }
                }
            }
        }

        int max = 0;
        KeyFrame *pKFmax = static_cast<KeyFrame *>(NULL);

        mvpLocalKeyFrames.clear();
        mvpLocalKeyFrames.reserve(3 * keyframeCounter.size());

        // All keyframes that observe a map point are included in the local map. Also check which keyframe shares most points
        for (map<KeyFrame *, int>::const_iterator it = keyframeCounter.begin(), itEnd = keyframeCounter.end(); it != itEnd; it++)
        {
            KeyFrame *pKF = it->first;

            if (pKF->isBad())
                continue;

            if (it->second > max)
            {
                max = it->second;
                pKFmax = pKF;
            }

            mvpLocalKeyFrames.push_back(pKF);
            pKF->mnTrackReferenceForFrame = mCurrentFrame.mnId;
        }

        // Include also some not-already-included keyframes that are neighbors to already-included keyframes
        for (vector<KeyFrame *>::const_iterator itKF = mvpLocalKeyFrames.begin(), itEndKF = mvpLocalKeyFrames.end(); itKF != itEndKF; itKF++)
        {
            // Limit the number of keyframes
            if (mvpLocalKeyFrames.size() > 80) // 80
                break;

            KeyFrame *pKF = *itKF;

            const vector<KeyFrame *> vNeighs = pKF->GetBestCovisibilityKeyFrames(10);

            for (vector<KeyFrame *>::const_iterator itNeighKF = vNeighs.begin(), itEndNeighKF = vNeighs.end(); itNeighKF != itEndNeighKF; itNeighKF++)
            {
                KeyFrame *pNeighKF = *itNeighKF;
                if (!pNeighKF->isBad())
                {
                    if (pNeighKF->mnTrackReferenceForFrame != mCurrentFrame.mnId)
                    {
                        mvpLocalKeyFrames.push_back(pNeighKF);
                        pNeighKF->mnTrackReferenceForFrame = mCurrentFrame.mnId;
                        break;
                    }
                }
            }

            const set<KeyFrame *> spChilds = pKF->GetChilds();
            for (set<KeyFrame *>::const_iterator sit = spChilds.begin(), send = spChilds.end(); sit != send; sit++)
            {
                KeyFrame *pChildKF = *sit;
                if (!pChildKF->isBad())
                {
                    if (pChildKF->mnTrackReferenceForFrame != mCurrentFrame.mnId)
                    {
                        mvpLocalKeyFrames.push_back(pChildKF);
                        pChildKF->mnTrackReferenceForFrame = mCurrentFrame.mnId;
                        break;
                    }
                }
            }

            KeyFrame *pParent = pKF->GetParent();
            if (pParent)
            {
                if (pParent->mnTrackReferenceForFrame != mCurrentFrame.mnId)
                {
                    mvpLocalKeyFrames.push_back(pParent);
                    pParent->mnTrackReferenceForFrame = mCurrentFrame.mnId;
                    break;
                }
            }
        }

        // Add 10 last temporal KFs (mainly for IMU)
        if ((mSensor == System::IMU_MONOCULAR || mSensor == System::IMU_STEREO || mSensor == System::IMU_RGBD) && mvpLocalKeyFrames.size() < 80)
        {
            KeyFrame *tempKeyFrame = mCurrentFrame.mpLastKeyFrame;

            const int Nd = 20;
            for (int i = 0; i < Nd; i++)
            {
                if (!tempKeyFrame)
                    break;
                if (tempKeyFrame->mnTrackReferenceForFrame != mCurrentFrame.mnId)
                {
                    mvpLocalKeyFrames.push_back(tempKeyFrame);
                    tempKeyFrame->mnTrackReferenceForFrame = mCurrentFrame.mnId;
                    tempKeyFrame = tempKeyFrame->mPrevKF;
                }
            }
        }

        if (pKFmax)
        {
            mpReferenceKF = pKFmax;
            mCurrentFrame.mpReferenceKF = mpReferenceKF;
        }
    }

    bool Tracking::Relocalization()
    {
        Verbose::PrintMess("Starting relocalization", Verbose::VERBOSITY_NORMAL);
        // Compute Bag of Words Vector
        mCurrentFrame.ComputeBoW();

        // Relocalization is performed when tracking is lost
        // Track Lost: Query KeyFrame Database for keyframe candidates for relocalisation
        vector<KeyFrame *> vpCandidateKFs = mpKeyFrameDB->DetectRelocalizationCandidates(&mCurrentFrame, mpAtlas->GetCurrentMap());

        if (vpCandidateKFs.empty())
        {
            Verbose::PrintMess("There are not candidates", Verbose::VERBOSITY_NORMAL);
            return false;
        }

        const int nKFs = vpCandidateKFs.size();

        // We perform first an ORB matching with each candidate
        // If enough matches are found we setup a PnP solver
        ORBmatcher matcher(0.75, true);

        vector<MLPnPsolver *> vpMLPnPsolvers;
        vpMLPnPsolvers.resize(nKFs);

        vector<vector<MapPoint *>> vvpMapPointMatches;
        vvpMapPointMatches.resize(nKFs);

        vector<bool> vbDiscarded;
        vbDiscarded.resize(nKFs);

        int nCandidates = 0;

        for (int i = 0; i < nKFs; i++)
        {
            KeyFrame *pKF = vpCandidateKFs[i];
            if (pKF->isBad())
                vbDiscarded[i] = true;
            else
            {
                int nmatches = matcher.SearchByBoW(pKF, mCurrentFrame, vvpMapPointMatches[i]);
                if (nmatches < 15)
                {
                    vbDiscarded[i] = true;
                    continue;
                }
                else
                {
                    MLPnPsolver *pSolver = new MLPnPsolver(mCurrentFrame, vvpMapPointMatches[i]);
                    pSolver->SetRansacParameters(0.99, 10, 300, 6, 0.5, 5.991); // This solver needs at least 6 points
                    vpMLPnPsolvers[i] = pSolver;
                    nCandidates++;
                }
            }
        }

        // Alternatively perform some iterations of P4P RANSAC
        // Until we found a camera pose supported by enough inliers
        bool bMatch = false;
        ORBmatcher matcher2(0.9, true);

        while (nCandidates > 0 && !bMatch)
        {
            for (int i = 0; i < nKFs; i++)
            {
                if (vbDiscarded[i])
                    continue;

                // Perform 5 Ransac Iterations
                vector<bool> vbInliers;
                int nInliers;
                bool bNoMore;

                MLPnPsolver *pSolver = vpMLPnPsolvers[i];
                Eigen::Matrix4f eigTcw;
                bool bTcw = pSolver->iterate(5, bNoMore, vbInliers, nInliers, eigTcw);

                // If Ransac reachs max. iterations discard keyframe
                if (bNoMore)
                {
                    vbDiscarded[i] = true;
                    nCandidates--;
                }

                // If a Camera Pose is computed, optimize
                if (bTcw)
                {
                    Sophus::SE3f Tcw(eigTcw);
                    mCurrentFrame.SetPose(Tcw);
                    // Tcw.copyTo(mCurrentFrame.mTcw);

                    set<MapPoint *> sFound;

                    const int np = vbInliers.size();

                    for (int j = 0; j < np; j++)
                    {
                        if (vbInliers[j])
                        {
                            mCurrentFrame.mvpMapPoints[j] = vvpMapPointMatches[i][j];
                            sFound.insert(vvpMapPointMatches[i][j]);
                        }
                        else
                            mCurrentFrame.mvpMapPoints[j] = NULL;
                    }

                    int nGood = Optimizer::PoseOptimization(&mCurrentFrame);

                    if (nGood < 10)
                        continue;

                    for (int io = 0; io < mCurrentFrame.N; io++)
                        if (mCurrentFrame.mvbOutlier[io])
                            mCurrentFrame.mvpMapPoints[io] = static_cast<MapPoint *>(NULL);

                    // If few inliers, search by projection in a coarse window and optimize again
                    if (nGood < 50)
                    {
                        int nadditional = matcher2.SearchByProjection(mCurrentFrame, vpCandidateKFs[i], sFound, 10, 100);

                        if (nadditional + nGood >= 50)
                        {
                            nGood = Optimizer::PoseOptimization(&mCurrentFrame);

                            // If many inliers but still not enough, search by projection again in a narrower window
                            // the camera has been already optimized with many points
                            if (nGood > 30 && nGood < 50)
                            {
                                sFound.clear();
                                for (int ip = 0; ip < mCurrentFrame.N; ip++)
                                    if (mCurrentFrame.mvpMapPoints[ip])
                                        sFound.insert(mCurrentFrame.mvpMapPoints[ip]);
                                nadditional = matcher2.SearchByProjection(mCurrentFrame, vpCandidateKFs[i], sFound, 3, 64);

                                // Final optimization
                                if (nGood + nadditional >= 50)
                                {
                                    nGood = Optimizer::PoseOptimization(&mCurrentFrame);

                                    for (int io = 0; io < mCurrentFrame.N; io++)
                                        if (mCurrentFrame.mvbOutlier[io])
                                            mCurrentFrame.mvpMapPoints[io] = NULL;
                                }
                            }
                        }
                    }

                    // If the pose is supported by enough inliers stop ransacs and continue
                    if (nGood >= 50)
                    {
                        bMatch = true;
                        break;
                    }
                }
            }
        }

        if (!bMatch)
        {
            return false;
        }
        else
        {
            mnLastRelocFrameId = mCurrentFrame.mnId;
            cout << "Relocalized!!" << endl;
            return true;
        }
    }

    void Tracking::Reset(bool bLocMap)
    {
        Verbose::PrintMess("System Reseting", Verbose::VERBOSITY_NORMAL);

        if (mpViewer)
        {
            mpViewer->RequestStop();
            while (!mpViewer->isStopped())
                usleep(3000);
        }

        // Reset Local Mapping
        if (!bLocMap)
        {
            Verbose::PrintMess("Reseting Local Mapper...", Verbose::VERBOSITY_NORMAL);
            mpLocalMapper->RequestReset();
            Verbose::PrintMess("done", Verbose::VERBOSITY_NORMAL);
        }

        // Reset Loop Closing
        Verbose::PrintMess("Reseting Loop Closing...", Verbose::VERBOSITY_NORMAL);
        mpLoopClosing->RequestReset();
        Verbose::PrintMess("done", Verbose::VERBOSITY_NORMAL);

        // Clear BoW Database
        Verbose::PrintMess("Reseting Database...", Verbose::VERBOSITY_NORMAL);
        mpKeyFrameDB->clear();
        Verbose::PrintMess("done", Verbose::VERBOSITY_NORMAL);

        // Clear Map (this erase MapPoints and KeyFrames)
        mpAtlas->clearAtlas();
        mpAtlas->CreateNewMap();
        if (mSensor == System::IMU_STEREO || mSensor == System::IMU_MONOCULAR || mSensor == System::IMU_RGBD)
            mpAtlas->SetInertialSensor();
        mnInitialFrameId = 0;

        KeyFrame::nNextId = 0;
        Frame::nNextId = 0;
        mState = NO_IMAGES_YET;

        mbReadyToInitializate = false;
        mbSetInit = false;

        mlRelativeFramePoses.clear();
        mlpReferences.clear();
        mlFrameTimes.clear();
        mlbLost.clear();
        mCurrentFrame = Frame();
        mnLastRelocFrameId = 0;
        mLastFrame = Frame();
        mpReferenceKF = static_cast<KeyFrame *>(NULL);
        mpLastKeyFrame = static_cast<KeyFrame *>(NULL);
        mvIniMatches.clear();

        if (mpViewer)
            mpViewer->Release();

        Verbose::PrintMess("   End reseting! ", Verbose::VERBOSITY_NORMAL);
    }

    void Tracking::ResetActiveMap(bool bLocMap)
    {
        Verbose::PrintMess("Active map Reseting", Verbose::VERBOSITY_NORMAL);
        if (mpViewer)
        {
            mpViewer->RequestStop();
            while (!mpViewer->isStopped())
                usleep(3000);
        }

        Map *pMap = mpAtlas->GetCurrentMap();

        if (!bLocMap)
        {
            Verbose::PrintMess("Reseting Local Mapper...", Verbose::VERBOSITY_VERY_VERBOSE);
            mpLocalMapper->RequestResetActiveMap(pMap);
            Verbose::PrintMess("done", Verbose::VERBOSITY_VERY_VERBOSE);
        }

        // Reset Loop Closing
        Verbose::PrintMess("Reseting Loop Closing...", Verbose::VERBOSITY_NORMAL);
        mpLoopClosing->RequestResetActiveMap(pMap);
        Verbose::PrintMess("done", Verbose::VERBOSITY_NORMAL);

        // Clear BoW Database
        Verbose::PrintMess("Reseting Database", Verbose::VERBOSITY_NORMAL);
        mpKeyFrameDB->clearMap(pMap); // Only clear the active map references
        Verbose::PrintMess("done", Verbose::VERBOSITY_NORMAL);

        // Clear Map (this erase MapPoints and KeyFrames)
        mpAtlas->clearMap();

        // KeyFrame::nNextId = mpAtlas->GetLastInitKFid();
        // Frame::nNextId = mnLastInitFrameId;
        mnLastInitFrameId = Frame::nNextId;
        // mnLastRelocFrameId = mnLastInitFrameId;
        mState = NO_IMAGES_YET; // NOT_INITIALIZED;

        mbReadyToInitializate = false;

        list<bool> lbLost;
        // lbLost.reserve(mlbLost.size());
        unsigned int index = mnFirstFrameId;
        cout << "mnFirstFrameId = " << mnFirstFrameId << endl;
        for (Map *pMap : mpAtlas->GetAllMaps())
        {
            if (pMap->GetAllKeyFrames().size() > 0)
            {
                if (index > pMap->GetLowerKFID())
                    index = pMap->GetLowerKFID();
            }
        }

        // cout << "First Frame id: " << index << endl;
        int num_lost = 0;
        cout << "mnInitialFrameId = " << mnInitialFrameId << endl;

        for (list<bool>::iterator ilbL = mlbLost.begin(); ilbL != mlbLost.end(); ilbL++)
        {
            if (index < mnInitialFrameId)
                lbLost.push_back(*ilbL);
            else
            {
                lbLost.push_back(true);
                num_lost += 1;
            }

            index++;
        }
        cout << num_lost << " Frames set to lost" << endl;

        mlbLost = lbLost;

        mnInitialFrameId = mCurrentFrame.mnId;
        mnLastRelocFrameId = mCurrentFrame.mnId;

        mCurrentFrame = Frame();
        mLastFrame = Frame();
        mpReferenceKF = static_cast<KeyFrame *>(NULL);
        mpLastKeyFrame = static_cast<KeyFrame *>(NULL);
        mvIniMatches.clear();

        mbVelocity = false;

        if (mpViewer)
            mpViewer->Release();

        Verbose::PrintMess("   End reseting! ", Verbose::VERBOSITY_NORMAL);
    }

    vector<MapPoint *> Tracking::GetLocalMapMPS()
    {
        return mvpLocalMapPoints;
    }

    void Tracking::ChangeCalibration(const string &strSettingPath)
    {
        cv::FileStorage fSettings(strSettingPath, cv::FileStorage::READ);
        float fx = fSettings["Camera.fx"];
        float fy = fSettings["Camera.fy"];
        float cx = fSettings["Camera.cx"];
        float cy = fSettings["Camera.cy"];

        mK_.setIdentity();
        mK_(0, 0) = fx;
        mK_(1, 1) = fy;
        mK_(0, 2) = cx;
        mK_(1, 2) = cy;

        cv::Mat K = cv::Mat::eye(3, 3, CV_32F);
        K.at<float>(0, 0) = fx;
        K.at<float>(1, 1) = fy;
        K.at<float>(0, 2) = cx;
        K.at<float>(1, 2) = cy;
        K.copyTo(mK);

        cv::Mat DistCoef(4, 1, CV_32F);
        DistCoef.at<float>(0) = fSettings["Camera.k1"];
        DistCoef.at<float>(1) = fSettings["Camera.k2"];
        DistCoef.at<float>(2) = fSettings["Camera.p1"];
        DistCoef.at<float>(3) = fSettings["Camera.p2"];
        const float k3 = fSettings["Camera.k3"];
        if (k3 != 0)
        {
            DistCoef.resize(5);
            DistCoef.at<float>(4) = k3;
        }
        DistCoef.copyTo(mDistCoef);

        mbf = fSettings["Camera.bf"];

        Frame::mbInitialComputations = true;
    }

    void Tracking::InformOnlyTracking(const bool &flag)
    {
        mbOnlyTracking = flag;
    }

    void Tracking::UpdateFrameIMU(const float s, const IMU::Bias &b, KeyFrame *pCurrentKeyFrame)
    {
        Map *pMap = pCurrentKeyFrame->GetMap();
        unsigned int index = mnFirstFrameId;
        list<ORB_SLAM3::KeyFrame *>::iterator lRit = mlpReferences.begin();
        list<bool>::iterator lbL = mlbLost.begin();
        for (auto lit = mlRelativeFramePoses.begin(), lend = mlRelativeFramePoses.end(); lit != lend; lit++, lRit++, lbL++)
        {
            if (*lbL)
                continue;

            KeyFrame *pKF = *lRit;

            while (pKF->isBad())
            {
                pKF = pKF->GetParent();
            }

            if (pKF->GetMap() == pMap)
            {
                (*lit).translation() *= s;
            }
        }

        mLastBias = b;

        mpLastKeyFrame = pCurrentKeyFrame;

        mLastFrame.SetNewBias(mLastBias);
        mCurrentFrame.SetNewBias(mLastBias);

        while (!mCurrentFrame.imuIsPreintegrated())
        {
            usleep(500);
        }

        if (mLastFrame.mnId == mLastFrame.mpLastKeyFrame->mnFrameId)
        {
            mLastFrame.SetImuPoseVelocity(mLastFrame.mpLastKeyFrame->GetImuRotation(),
                                          mLastFrame.mpLastKeyFrame->GetImuPosition(),
                                          mLastFrame.mpLastKeyFrame->GetVelocity());
        }
        else
        {
            const Eigen::Vector3f Gz(0, 0, -IMU::GRAVITY_VALUE);
            const Eigen::Vector3f twb1 = mLastFrame.mpLastKeyFrame->GetImuPosition();
            const Eigen::Matrix3f Rwb1 = mLastFrame.mpLastKeyFrame->GetImuRotation();
            const Eigen::Vector3f Vwb1 = mLastFrame.mpLastKeyFrame->GetVelocity();
            float t12 = mLastFrame.mpImuPreintegrated->dT;

            mLastFrame.SetImuPoseVelocity(IMU::NormalizeRotation(Rwb1 * mLastFrame.mpImuPreintegrated->GetUpdatedDeltaRotation()),
                                          twb1 + Vwb1 * t12 + 0.5f * t12 * t12 * Gz + Rwb1 * mLastFrame.mpImuPreintegrated->GetUpdatedDeltaPosition(),
                                          Vwb1 + Gz * t12 + Rwb1 * mLastFrame.mpImuPreintegrated->GetUpdatedDeltaVelocity());
        }

        if (mCurrentFrame.mpImuPreintegrated)
        {
            const Eigen::Vector3f Gz(0, 0, -IMU::GRAVITY_VALUE);

            const Eigen::Vector3f twb1 = mCurrentFrame.mpLastKeyFrame->GetImuPosition();
            const Eigen::Matrix3f Rwb1 = mCurrentFrame.mpLastKeyFrame->GetImuRotation();
            const Eigen::Vector3f Vwb1 = mCurrentFrame.mpLastKeyFrame->GetVelocity();
            float t12 = mCurrentFrame.mpImuPreintegrated->dT;

            mCurrentFrame.SetImuPoseVelocity(IMU::NormalizeRotation(Rwb1 * mCurrentFrame.mpImuPreintegrated->GetUpdatedDeltaRotation()),
                                             twb1 + Vwb1 * t12 + 0.5f * t12 * t12 * Gz + Rwb1 * mCurrentFrame.mpImuPreintegrated->GetUpdatedDeltaPosition(),
                                             Vwb1 + Gz * t12 + Rwb1 * mCurrentFrame.mpImuPreintegrated->GetUpdatedDeltaVelocity());
        }

        mnFirstImuFrameId = mCurrentFrame.mnId;
    }

    void Tracking::NewDataset()
    {
        mnNumDataset++;
    }

    int Tracking::GetNumberDataset()
    {
        return mnNumDataset;
    }

    int Tracking::GetMatchesInliers()
    {
        return mnMatchesInliers;
    }

    void Tracking::SaveSubTrajectory(string strNameFile_frames, string strNameFile_kf, string strFolder)
    {
        mpSystem->SaveTrajectoryEuRoC(strFolder + strNameFile_frames);
        // mpSystem->SaveKeyFrameTrajectoryEuRoC(strFolder + strNameFile_kf);
    }

    void Tracking::SaveSubTrajectory(string strNameFile_frames, string strNameFile_kf, Map *pMap)
    {
        mpSystem->SaveTrajectoryEuRoC(strNameFile_frames, pMap);
        if (!strNameFile_kf.empty())
            mpSystem->SaveKeyFrameTrajectoryEuRoC(strNameFile_kf, pMap);
    }

    float Tracking::GetImageScale()
    {
        return mImageScale;
    }

#ifdef REGISTER_LOOP
    void Tracking::RequestStop()
    {
        unique_lock<mutex> lock(mMutexStop);
        mbStopRequested = true;
    }

    bool Tracking::Stop()
    {
        unique_lock<mutex> lock(mMutexStop);
        if (mbStopRequested && !mbNotStop)
        {
            mbStopped = true;
            cout << "Tracking STOP" << endl;
            return true;
        }

        return false;
    }

    bool Tracking::stopRequested()
    {
        unique_lock<mutex> lock(mMutexStop);
        return mbStopRequested;
    }

    bool Tracking::isStopped()
    {
        unique_lock<mutex> lock(mMutexStop);
        return mbStopped;
    }

    void Tracking::Release()
    {
        unique_lock<mutex> lock(mMutexStop);
        mbStopped = false;
        mbStopRequested = false;
    }
#endif

// **新增：独立的全局光流动态检测**
// 目标：在YOLO检测之前，先用光流标记所有动态特征点
// 原理：对比当前帧和上一帧的所有特征点，检测异常运动
void Tracking::GlobalFlowDynamicDetection()
{
    // 安全检查
    if(mLastImGray.empty() || mImGray.empty() || mLastFrame.N == 0)
        return;
    
    // 1. 光流追踪：上一帧特征点 → 当前帧位置
    vector<cv::Point2f> vPtsPrev, vPtsCurr;
    vPtsPrev.reserve(mLastFrame.N);
    
    for(int i=0; i<mLastFrame.N; i++) {
        vPtsPrev.push_back(mLastFrame.mvKeysUn[i].pt);
    }
    
    vector<uchar> status;
    vector<float> error;
    cv::calcOpticalFlowPyrLK(mLastImGray, mImGray, 
                             vPtsPrev, vPtsCurr, 
                             status, error,
                             cv::Size(21, 21), 3);
    
    // 2. 计算全局光流中位数（背景运动）
    vector<float> vFlowU, vFlowV;
    vFlowU.reserve(mLastFrame.N);
    vFlowV.reserve(mLastFrame.N);
    
    for(size_t i=0; i<vPtsPrev.size(); i++) {
        if(!status[i]) continue;
        
        float u = vPtsCurr[i].x - vPtsPrev[i].x;
        float v = vPtsCurr[i].y - vPtsPrev[i].y;
        vFlowU.push_back(u);
        vFlowV.push_back(v);
    }
    
    if(vFlowU.size() < 10) {
        std::cout << "GlobalFlow: Too few points (" << vFlowU.size() << "), skipping" << std::endl;
        return;
    }
    
    std::sort(vFlowU.begin(), vFlowU.end());
    std::sort(vFlowV.begin(), vFlowV.end());
    
    float median_u = vFlowU[vFlowU.size() / 2];
    float median_v = vFlowV[vFlowV.size() / 2];
    
    // 3. 检测outliers并标记为动态点
    float th_flow = 3.0f;  // 阈值：3像素
    int nDynamicPoints = 0;
    
    for(size_t i=0; i<vPtsPrev.size(); i++) {
        if(!status[i] || i >= mLastFrame.mvpMapPoints.size()) continue;
        if(!mLastFrame.mvpMapPoints[i]) continue;
        
        float u = vPtsCurr[i].x - vPtsPrev[i].x;
        float v = vPtsCurr[i].y - vPtsPrev[i].y;
        
        float diff_u = u - median_u;
        float diff_v = v - median_v;
        float diff = sqrt(diff_u*diff_u + diff_v*diff_v);
        
        if(diff > th_flow) {
            // 标记为动态点 - 直接设置为NULL，在tracking中不再使用
            mLastFrame.mvpMapPoints[i]->SetBadFlag();
            mLastFrame.mvpMapPoints[i] = static_cast<MapPoint*>(NULL);
            nDynamicPoints++;
        }
    }
    
    // NOTE: Per-frame stdout is very expensive; gate behind YDM_VERBOSE.
    const bool ydm_verbose = (std::getenv("YDM_VERBOSE") && std::string(std::getenv("YDM_VERBOSE")) == "1");
    if (ydm_verbose)
        std::cout << "GlobalFlow: Detected " << nDynamicPoints << " dynamic points (out of "
                  << vPtsPrev.size() << "), bg_flow=(" << median_u << "," << median_v << ")" << std::endl;
}

// **新增核心：利用深度图优化 YOLO Mask**
cv::Mat Tracking::GetRefinedDynamicMask(const cv::Mat& imDepth, const std::vector<cv::Rect>& boxes) {
    cv::Mat final_mask = cv::Mat::zeros(imDepth.size(), CV_8UC1);  // Updated to CV_8UC1
    
    // 如果没有检测框，直接返回空Mask
    if(boxes.empty()) return final_mask;

    for(const auto& box : boxes) {
        // 1. 边界检查，防止越界
        cv::Rect safe_box = box & cv::Rect(0, 0, imDepth.cols, imDepth.rows);
        if(safe_box.area() <= 0) continue;

        cv::Mat roi_depth = imDepth(safe_box);
        
        // 2. 计算物体主体深度 (中位数)
        std::vector<float> depths;
        depths.reserve(roi_depth.total());
        
        // 优化：使用迭代器遍历
        for(int i=0; i<roi_depth.rows; i++) {
            const float* ptr = roi_depth.ptr<float>(i);
            for(int j=0; j<roi_depth.cols; j++) {
                float d = ptr[j];
                // 过滤无效深度和过远深度 (0.1m - 15.0m)
                if(d > 0.1f && d < 15.0f) {
                    depths.push_back(d);
                }
            }
        }
        
        if(depths.empty()) continue; // 无有效深度
        
        // 找中位数
        size_t n = depths.size() / 2;
        std::nth_element(depths.begin(), depths.begin() + n, depths.end());
        float median_d = depths[n];
        
        // 3. 深度阈值切割生成 Mask
        cv::Mat roi_mask = cv::Mat::zeros(safe_box.size(), CV_8UC1);
        float tolerance = 0.5f; // 深度容差 +/- 0.5m
        
        for(int i=0; i<roi_depth.rows; i++) {
            const float* ptr_d = roi_depth.ptr<float>(i);
            uchar* ptr_m = roi_mask.ptr<uchar>(i);
            for(int j=0; j<roi_depth.cols; j++) {
                if(std::abs(ptr_d[j] - median_d) < tolerance) {
                    ptr_m[j] = 255;
                }
            }
        }
        
        // 4. 将 ROI mask 填回全图 mask (使用 OR 操作以支持重叠框)
        cv::Mat final_roi = final_mask(safe_box);
        cv::bitwise_or(final_roi, roi_mask, final_roi);
    }

    // 5. 全局形态学膨胀 (解决边缘特征点残留)
    cv::Mat dilated_mask;
    // 使用 7x7 核进行膨胀
    cv::dilate(final_mask, dilated_mask, 
               cv::getStructuringElement(cv::MORPH_RECT, cv::Size(7, 7)));
               
    return dilated_mask;
}

// **新增优化：综合验证与特征剔除**
void Tracking::VerifyAndPruneFeatures() {
    // Profiling / logging knobs (env vars)
    // - YDM_PROFILE=0: disable per-stage timing logs (enabled by default)
    // - YDM_PROFILE_EVERY=N: print every N frames (default 1)
    // - YDM_VERBOSE=1: enable verbose logs
    // - YDM_FLOW_BOX_LOG=1: print per-box flow decisions (can be very slow)
    const char *env_profile = std::getenv("YDM_PROFILE");
    const bool ydm_profile = !(env_profile && std::string(env_profile) == "0");
    int ydm_profile_every = 1;
    if (const char* env_pe = std::getenv("YDM_PROFILE_EVERY"))
    {
        try { ydm_profile_every = std::stoi(std::string(env_pe)); } catch (...) { ydm_profile_every = 1; }
        if (ydm_profile_every < 1) ydm_profile_every = 1;
    }
    const bool ydm_verbose = (std::getenv("YDM_VERBOSE") && std::string(std::getenv("YDM_VERBOSE")) == "1");
    const bool ydm_flow_box_log = (std::getenv("YDM_FLOW_BOX_LOG") && std::string(std::getenv("YDM_FLOW_BOX_LOG")) == "1");

    const auto ydm_t0_all = std::chrono::steady_clock::now();
    auto ydm_t = ydm_t0_all;
    auto ydm_ms_since = [](const std::chrono::steady_clock::time_point &a,
                           const std::chrono::steady_clock::time_point &b) -> double {
        return std::chrono::duration_cast<std::chrono::duration<double, std::milli>>(b - a).count();
    };

    // Windowed profiling accumulators (counted per Verify call, not per frame ID)
    static int ydm_verify_calls = 0;
    static int ydm_window_calls = 0;
    static double ydm_sum_depth_ms = 0.0;
    static double ydm_sum_yolo_union_ms = 0.0;
    static double ydm_sum_refined_ms = 0.0;
    static double ydm_sum_box_ms = 0.0;
    static double ydm_sum_mask_save_ms = 0.0;
    static double ydm_sum_prune_ms = 0.0;
    static double ydm_sum_total_ms = 0.0;

    // 0. 安全检查
    // 使用 Tracking 类的 mImDepth 成员，而不是 Frame 的
    // 该函数默认复用 TrackWithMotionModel() 中缓存的 residual（FlowNet + ego compensation）。
    if(mImDepth.empty() || mpORBextractorLeft->mvDynamicArea.empty())
        return;
    const bool has_residual_cache = (!mCurrResidual.empty() && !mCurrResidualValidMask.empty() &&
                                    mCurrResidual.type() == CV_32FC1 && mCurrResidualValidMask.type() == CV_8UC1 &&
                                    mCurrResidual.size() == mImGray.size() &&
                                    mCurrResidualFrameId == mCurrentFrame.mnId);

    // 重要：mImDepth 在 GrabImageRGBD() 里保存的是原始深度图（常见为 CV_16U），
    // 但 GetRefinedDynamicMask() 按 CV_32F(米) 读取深度并做阈值切割。
    // 如果不先转换，会导致 depths 为空/median 深度异常，refined_mask 结果全黑。
    cv::Mat depth_for_mask = mImDepth;
    if ((fabs(mDepthMapFactor - 1.0f) > 1e-5) || depth_for_mask.type() != CV_32F)
        depth_for_mask.convertTo(depth_for_mask, CV_32F, mDepthMapFactor);

    const auto ydm_t_depth = std::chrono::steady_clock::now();
    const double ydm_ms_depth = ydm_ms_since(ydm_t, ydm_t_depth);
    ydm_t = ydm_t_depth;

    // Step A0: 以 YOLO 分割 mask 作为“保底动态区域”
    // 说明：仅用 box 内深度中位数切割可能会选到背景深度（尤其当box里背景像素占比更大），
    // 从而把真正的动态物体漏掉。这里先把 YOLO mask 做并集，后续再结合深度 refined。
    cv::Mat yolo_union = cv::Mat::zeros(depth_for_mask.size(), CV_8UC1);
    if (!mpORBextractorLeft->mvDynamicMask.empty())
    {
        for (const cv::Mat &m : mpORBextractorLeft->mvDynamicMask)
        {
            if (m.empty())
                continue;

            cv::Mat m1;
            if (m.channels() == 1)
                m1 = m;
            else
                cv::cvtColor(m, m1, cv::COLOR_BGR2GRAY);

            if (m1.type() != CV_8UC1)
                m1.convertTo(m1, CV_8UC1);

            if (m1.size() != yolo_union.size())
            {
                cv::Mat resized;
                cv::resize(m1, resized, yolo_union.size(), 0, 0, cv::INTER_NEAREST);
                m1 = resized;
            }

            cv::bitwise_or(yolo_union, m1, yolo_union);
        }
    }

    const auto ydm_t_yolo_union = std::chrono::steady_clock::now();
    const double ydm_ms_yolo_union = ydm_ms_since(ydm_t, ydm_t_yolo_union);
    ydm_t = ydm_t_yolo_union;

    // Step A: 生成基于深度的精准 Mask
    // 注意：我们需要将 cv::Rect2f 转换为 cv::Rect 才能传递给 GetRefinedDynamicMask
    // 或者直接修改 GetRefinedDynamicMask 接受 vector<Rect2i>
    // 这里我们做个简单的适配
    std::vector<cv::Rect> vRects;
    for(const auto& r : mpORBextractorLeft->mvDynamicArea) {
        vRects.push_back(r);
    }

    cv::Mat depth_refined_mask = GetRefinedDynamicMask(depth_for_mask, vRects);

    const auto ydm_t_refined = std::chrono::steady_clock::now();
    const double ydm_ms_refined = ydm_ms_since(ydm_t, ydm_t_refined);
    ydm_t = ydm_t_refined;

    // 组合 mask：优先保留已有的 refined mask（例如 DynamicObjectCleaner 已经写入 mCurrentFrame.mvDynamicMask），
    // 再与深度 refined + YOLO union 做并集，避免丢弃任意一路的动态判断。
    cv::Mat refined_mask = cv::Mat::zeros(depth_for_mask.size(), CV_8UC1);
    if (!mCurrentFrame.mvDynamicMask.empty() && !mCurrentFrame.mvDynamicMask[0].empty())
    {
        cv::Mat m0 = mCurrentFrame.mvDynamicMask[0];
        if (m0.size() != refined_mask.size())
            cv::resize(m0, m0, refined_mask.size(), 0, 0, cv::INTER_NEAREST);
        if (m0.type() != CV_8UC1)
            m0.convertTo(m0, CV_8UC1);
        refined_mask = m0.clone();
    }
    cv::bitwise_or(refined_mask, depth_refined_mask, refined_mask);
    if (!yolo_union.empty())
        cv::bitwise_or(refined_mask, yolo_union, refined_mask);

    cv::Mat refined_mask_1 = refined_mask;
    // Step B: 预处理 - 基于光流判断每个 YOLO 框的动静状态
    // 我们统计每个框内的特征点，看有多少比例是"显著运动"的 (diff > 2.5)
    // 如果大部分是静止的，说明这个框（例如坐着的人）是静态物体，我们需要将其从 Mask 中移除，
    // 以便在 PCD 建图中保留，并在 Tracking 中保留特征点。

    struct BoxStats {
        int total = 0;
        int moving = 0;
        std::vector<float> diffs;
    };
    std::map<int, BoxStats> box_stats_map; // Index in mvDynamicArea -> Stats

    // 准备运动证据：默认使用 residual cache（FlowNet + ego compensation）
    const bool bCheckMotion = has_residual_cache;

    // 使用 residual 阈值做“框内是否运动”的判定
    const float kMovingResidualTh = (has_residual_cache && mCurrResidualThr > 1e-6f) ? mCurrResidualThr : 2.5f;
    const int   kMinPtsPerBox = 20;

            // 可配置：光流动态比例阈值，默认 0.50
            // 用法：运行前设置环境变量 `YDM_FLOW_DYNAMIC_RATIO_TH=0.5`
            float kDynamicRatioTh = 0.50f;
            if (const char *env_ratio = std::getenv("YDM_FLOW_DYNAMIC_RATIO_TH"))
            {
                try
                {
                    kDynamicRatioTh = std::stof(std::string(env_ratio));
                }
                catch (...)
                {
                    kDynamicRatioTh = 0.50f;
                }
                if (kDynamicRatioTh < 0.0f) kDynamicRatioTh = 0.0f;
                if (kDynamicRatioTh > 1.0f) kDynamicRatioTh = 1.0f;
            }

    for(int i=0; i<mCurrentFrame.N; i++) {
        if (!bCheckMotion) break;
        const int x = cvRound(mCurrentFrame.mvKeysUn[i].pt.x);
        const int y = cvRound(mCurrentFrame.mvKeysUn[i].pt.y);
        if (x < 0 || y < 0 || x >= mCurrResidual.cols || y >= mCurrResidual.rows)
            continue;
        if (!mCurrResidualValidMask.at<uchar>(y, x))
            continue;
        const float diff = mCurrResidual.at<float>(y, x);

                // 检查点属于哪个框
                // mpORBextractorLeft->mvDynamicArea 是原始 YOLO 框
                for(size_t k=0; k < mpORBextractorLeft->mvDynamicArea.size(); k++) {
                    if(mpORBextractorLeft->mvDynamicArea[k].contains(mCurrentFrame.mvKeysUn[i].pt)) {
                        box_stats_map[k].total++;
                        box_stats_map[k].diffs.push_back(diff);
                        if(diff > kMovingResidualTh) {
                            box_stats_map[k].moving++;
                        }
                        // 一个点只归属第一个匹配的框 (简化)
                        break;
                    }
                }
            }

            // 根据统计结果更新 Mask
            // - ratio > 0.5: 保留该框为动态（mask 不清除），后续 mask 内点全部剔除
            // - ratio <= 0.5: 认为该框整体偏静态，清除该框区域的 mask（保留点/点云）
            for(std::map<int, BoxStats>::const_iterator it = box_stats_map.begin(); it != box_stats_map.end(); ++it) {
                const int k = it->first;
                const BoxStats &stats = it->second;
                if(stats.total < kMinPtsPerBox) {
                    continue; // 证据不足：默认不清mask（更保守）
                }

                const float ratio = (float)stats.moving / (float)stats.total;
                if(ratio <= kDynamicRatioTh) {
                    if (ydm_flow_box_log || ydm_verbose)
                        std::cout << "Reverting Box (static by flow): Box " << k
                                  << " (Moving: " << stats.moving << "/" << stats.total
                                  << " Ratio: " << ratio << ")" << std::endl;

                    cv::Rect safe = mpORBextractorLeft->mvDynamicArea[k] & cv::Rect(0, 0, refined_mask.cols, refined_mask.rows);
                    if(safe.area() > 0) {
                        refined_mask(safe).setTo(0);
                    }
                } else {
                    if (ydm_flow_box_log || ydm_verbose)
                        std::cout << "Keeping Box (dynamic by flow): Box " << k
                                  << " (Moving: " << stats.moving << "/" << stats.total
                                  << " Ratio: " << ratio << ")" << std::endl;
                }
            }

    const auto ydm_t_box = std::chrono::steady_clock::now();
    const double ydm_ms_box = ydm_ms_since(ydm_t, ydm_t_box);
    ydm_t = ydm_t_box;


    // 更新 Frame 中的 Mask，供后续建图使用 (此时 mask 已经被"平反"逻辑修正过)
    mCurrentFrame.mvDynamicMask.clear();
    mCurrentFrame.mvDynamicMask.push_back(refined_mask);

    // 可选：保存 refined_mask（用于调试点云/动态剔除效果）
    // 用法：在运行前设置环境变量 `YDM_SAVE_REFINED_MASK=1`
    // 额外：可设置 `YDM_SAVE_REFINED_MASK_EVERY=N` 控制每 N 帧保存一次（例如 N=2 表示每两帧保存）
    if (const char *env = std::getenv("YDM_SAVE_REFINED_MASK"))
    {
        if (std::string(env) == "1")
        {
            try
            {
                int save_every = 1;
                if (const char *env_every = std::getenv("YDM_SAVE_REFINED_MASK_EVERY"))
                {
                    try
                    {
                        save_every = std::max(1, std::stoi(std::string(env_every)));
                    }
                    catch (...)
                    {
                        save_every = 1;
                    }
                }

                // 只在需要保存的帧才执行保存操作
                if (save_every <= 1 || (mCurrentFrame.mnId % save_every) == 0)
                {
                    // 确保 mask 目录存在
                    static bool sMaskDirReady = false;
                    if (!sMaskDirReady)
                    {
                        const int rc = mkdir("mask", 0755);
                        if (rc == 0 || errno == EEXIST)
                            sMaskDirReady = true;
                    }

                    if (sMaskDirReady)
                    {
                        std::ostringstream oss;
                        oss << "mask/refined_mask_" << std::setw(6) << std::setfill('0') << mCurrentFrame.mnId << ".png";
                        const bool ok = cv::imwrite(oss.str(), refined_mask_1);
                        if (!ok)
                        {
                            std::cerr << "WARNING: Failed to save refined mask to '" << oss.str() << "'" << std::endl;
                        }
                    }
                }
            }
            catch (const cv::Exception &e)
            {
                std::cerr << "WARNING: Exception while saving refined mask: " << e.what() << std::endl;
            }
        }
    }

    const auto ydm_t_mask_save = std::chrono::steady_clock::now();
    const double ydm_ms_mask_save = ydm_ms_since(ydm_t, ydm_t_mask_save);
    ydm_t = ydm_t_mask_save;


    // Step C: 遍历特征点进行剔除 (使用更新后的 Mask)
    int nPrunedMask = 0;
    int nPrunedGeom = 0;

    // Adaptive Pruning Strategy Loop
    float currentGeoThreshold = 5.0f; // Relaxed default from 2.5f to 5.0f to prevent jitter
    bool bSafetyTriggered = false;

    // We might need to run this twice if safety mechanism is triggered
    for(int attempt=0; attempt<2; attempt++) {
        nPrunedMask = 0;
        nPrunedGeom = 0;
        
        // Reset outliers first if this is a retry
        if(attempt > 0) {
             for(int i=0; i<mCurrentFrame.N; i++) {
                 mCurrentFrame.mvbOutlier[i] = false;
                 // Don't reset class_id here, will overwrite below
             }
        }

        for(int i=0; i<mCurrentFrame.N; i++) {
            const cv::Point2f &pt = mCurrentFrame.mvKeysUn[i].pt;
            int x = cvRound(pt.x);
            int y = cvRound(pt.y);
            
            // 边界检查
            if(x < 0 || x >= refined_mask.cols || y < 0 || y >= refined_mask.rows)
                continue;

            // Condition A: 落在 Refined Mask 内
            // 注意：因为上面已经把静态框涂黑了，所以这里的 in_mask 已经是 false 了
            bool in_mask = refined_mask.at<uchar>(y, x) > 0;
            
            bool should_prune = false;
            float diff = 0.0f;

            // 策略修正：
            // - 光流只用于“平反整框”(Step B)，不要用于在 mask 内逐点放行。
            //   否则动态物体在某些帧里会因为 diff 小（例如与相机运动接近/光流抖动/跟踪失败）而被保留。
            // - 因此：只要点仍在 refined_mask 内，一律剔除。
            if (in_mask) {
                should_prune = true;
            } else if (bCheckMotion) {
                if (x >= 0 && y >= 0 && x < mCurrResidual.cols && y < mCurrResidual.rows &&
                    mCurrResidualValidMask.at<uchar>(y, x))
                {
                    diff = mCurrResidual.at<float>(y, x);
                    // Mask 外的点，仅剔除“显著运动”的点
                    if (diff > currentGeoThreshold)
                        should_prune = true;
                }
            }

            if (should_prune) {
                mCurrentFrame.mvbOutlier[i] = true;
                mCurrentFrame.mvKeys[i].class_id = -99;
                
                if(in_mask) nPrunedMask++;
                else nPrunedGeom++; // Simplified stats
              } else {
                  // 不要覆盖先前模块（如 DynamicObjectCleaner）已经标记的 outlier
                  // 这里仅在当前未被标记时保持 false。
                  if (!mCurrentFrame.mvbOutlier[i])
                     mCurrentFrame.mvbOutlier[i] = false;
                  // 同理，不强制重置 class_id
              }
        }
        
        int nRemaining = mCurrentFrame.N - (nPrunedMask + nPrunedGeom);
        
        // Safety Check: If retention is less than 50%, relax threshold and retry
        // Note: This mainly affects the !in_mask points, as in_mask points have fixed 2.5px threshold
        if(attempt == 0 && nRemaining < (mCurrentFrame.N / 2)) {
            if (ydm_verbose)
                std::cout << "WARNING: Excessive Pruning! Remaining: " << nRemaining << "/" << mCurrentFrame.N 
                          << " (<50%). Relaxing Geometric Threshold to 10.0px." << std::endl;
            currentGeoThreshold = 10.0f;
            bSafetyTriggered = true;
            continue; // Retry loop
        }
        
        if (ydm_verbose)
            std::cout << "Dynamic Pruning: " << nPrunedMask << " by Mask, " << nPrunedGeom << " by Geometry. "
                      << "Remaining: " << nRemaining << "/" << mCurrentFrame.N << std::endl;
                  
        if(attempt > 0) break; // Finished retry
        if(!bSafetyTriggered) break; // Finished first try successfully
    }

    // Windowed summary (averaged). Printed every N Verify calls.
    const auto ydm_t_end = std::chrono::steady_clock::now();
    const double ydm_ms_prune = ydm_ms_since(ydm_t, ydm_t_end);
    const double ydm_ms_total = ydm_ms_since(ydm_t0_all, ydm_t_end);

    ydm_verify_calls++;
    ydm_window_calls++;
    ydm_sum_depth_ms += ydm_ms_depth;
    ydm_sum_yolo_union_ms += ydm_ms_yolo_union;
    ydm_sum_refined_ms += ydm_ms_refined;
    ydm_sum_box_ms += ydm_ms_box;
    ydm_sum_mask_save_ms += ydm_ms_mask_save;
    ydm_sum_prune_ms += ydm_ms_prune;
    ydm_sum_total_ms += ydm_ms_total;

    if (ydm_profile && ydm_window_calls >= ydm_profile_every)
    {
        const double n = static_cast<double>(ydm_window_calls);
        const double avg_total = ydm_sum_total_ms / n;
        const double fps = (avg_total > 1e-9) ? (1000.0 / avg_total) : 0.0;

        std::cout << "[YDM][VERIFY] call=" << std::setw(6) << ydm_verify_calls
                  << " every=" << ydm_profile_every
                  << " count=" << ydm_window_calls
                  << " lastF=" << std::setw(6) << mCurrentFrame.mnId
                  << " | Depth=" << std::setw(6) << std::fixed << std::setprecision(2) << (ydm_sum_depth_ms / n) << "ms"
                  << " YoloUnion=" << std::setw(6) << (ydm_sum_yolo_union_ms / n) << "ms"
                  << " Refined=" << std::setw(6) << (ydm_sum_refined_ms / n) << "ms"
                  << " Box=" << std::setw(6) << (ydm_sum_box_ms / n) << "ms"
                  << " MaskSave=" << std::setw(6) << (ydm_sum_mask_save_ms / n) << "ms"
                  << " Prune=" << std::setw(6) << (ydm_sum_prune_ms / n) << "ms"
                  << " || TOTAL=" << std::setw(7) << avg_total << "ms"
                  << " FPS=" << std::setw(6) << std::setprecision(2) << fps
                  << std::endl;

        ydm_window_calls = 0;
        ydm_sum_depth_ms = 0.0;
        ydm_sum_yolo_union_ms = 0.0;
        ydm_sum_refined_ms = 0.0;
        ydm_sum_box_ms = 0.0;
        ydm_sum_mask_save_ms = 0.0;
        ydm_sum_prune_ms = 0.0;
        ydm_sum_total_ms = 0.0;
    }

    // ========== Debug Visualization for Viewer (YDM_VIEW_DEBUG=1) ==========
    // Generate three pseudocolor images: Raw YOLO mask, Refined mask, Residual heatmap
    // Push to Viewer every N frames (YDM_VIEW_DEBUG_EVERY, default=3)
    if (mpViewer)
    {
        static bool sDebugChecked = false;
        static bool sDebugEnabled = false;
        static int sDebugEvery = 3;
        static float sResidualMax = 10.0f;
        static float sFlowMax = 20.0f;
        if (!sDebugChecked)
        {
            sDebugChecked = true;
            sDebugEnabled = (std::getenv("YDM_VIEW_DEBUG") && std::string(std::getenv("YDM_VIEW_DEBUG")) == "1");
            if (const char* env = std::getenv("YDM_VIEW_DEBUG_EVERY"))
            {
                try { sDebugEvery = std::max(1, std::stoi(std::string(env))); } catch (...) { sDebugEvery = 3; }
            }
            if (const char* env = std::getenv("YDM_VIEW_RESIDUAL_MAX"))
            {
                try { sResidualMax = std::max(1.0f, std::stof(std::string(env))); } catch (...) { sResidualMax = 10.0f; }
            }
            if (const char* env = std::getenv("YDM_VIEW_FLOW_MAX"))
            {
                try { sFlowMax = std::max(1.0f, std::stof(std::string(env))); } catch (...) { sFlowMax = 20.0f; }
            }
        }

        if (sDebugEnabled && (static_cast<int>(mCurrentFrame.mnId) % sDebugEvery == 0))
        {
            const cv::Size targetSize = mImGray.size();

            // 1) Raw YOLO mask (pseudocolor from yolo_union)
            cv::Mat rawYoloVis;
            if (!yolo_union.empty())
            {
                cv::Mat yolo_resized = yolo_union;
                if (yolo_union.size() != targetSize)
                    cv::resize(yolo_union, yolo_resized, targetSize, 0, 0, cv::INTER_NEAREST);
                cv::applyColorMap(yolo_resized, rawYoloVis, cv::COLORMAP_JET);
            }
            else
            {
                rawYoloVis = cv::Mat::zeros(targetSize, CV_8UC3);
            }

            // 2) Refined mask (pseudocolor from final refined_mask)
            cv::Mat refinedVis;
            if (!refined_mask.empty())
            {
                cv::Mat ref_resized = refined_mask;
                if (refined_mask.size() != targetSize)
                    cv::resize(refined_mask, ref_resized, targetSize, 0, 0, cv::INTER_NEAREST);
                cv::applyColorMap(ref_resized, refinedVis, cv::COLORMAP_JET);
            }
            else
            {
                refinedVis = cv::Mat::zeros(targetSize, CV_8UC3);
            }

            // 3) Residual heatmap (pseudocolor from mCurrResidual)
            cv::Mat residualVis;
            if (has_residual_cache && !mCurrResidual.empty())
            {
                cv::Mat res_resized = mCurrResidual;
                if (mCurrResidual.size() != targetSize)
                    cv::resize(mCurrResidual, res_resized, targetSize, 0, 0, cv::INTER_LINEAR);

                // Normalize to [0, 255] using configurable max
                cv::Mat res_norm;
                res_resized.convertTo(res_norm, CV_8UC1, 255.0 / sResidualMax, 0);
                cv::applyColorMap(res_norm, residualVis, cv::COLORMAP_JET);
            }
            else
            {
                residualVis = cv::Mat::zeros(targetSize, CV_8UC3);
            }

            // 4) Refined motion mask (binary -> pseudocolor)
            cv::Mat refinedMotionVis;
            if (has_residual_cache && !mCurrResidualMotionMask.empty())
            {
                cv::Mat mm_resized = mCurrResidualMotionMask;
                if (mCurrResidualMotionMask.size() != targetSize)
                    cv::resize(mCurrResidualMotionMask, mm_resized, targetSize, 0, 0, cv::INTER_NEAREST);
                cv::applyColorMap(mm_resized, refinedMotionVis, cv::COLORMAP_JET);
            }
            else
            {
                refinedMotionVis = cv::Mat::zeros(targetSize, CV_8UC3);
            }

            // 5) Raw flow magnitude (mCurrFlowPrevToCurr -> magnitude -> pseudocolor)
            cv::Mat rawFlowVis;
            if (has_residual_cache && !mCurrFlowPrevToCurr.empty())
            {
                cv::Mat flow_resized = mCurrFlowPrevToCurr;
                if (mCurrFlowPrevToCurr.size() != targetSize)
                    cv::resize(mCurrFlowPrevToCurr, flow_resized, targetSize, 0, 0, cv::INTER_LINEAR);

                cv::Mat mag(targetSize, CV_32FC1, cv::Scalar(0));
                for (int y = 0; y < targetSize.height; ++y)
                {
                    const cv::Vec2f* fptr = flow_resized.ptr<cv::Vec2f>(y);
                    float* mptr = mag.ptr<float>(y);
                    for (int x = 0; x < targetSize.width; ++x)
                    {
                        const cv::Vec2f v = fptr[x];
                        const float m = std::sqrt(v[0] * v[0] + v[1] * v[1]);
                        mptr[x] = std::isfinite(m) ? m : 0.0f;
                    }
                }

                cv::Mat mag_u8;
                mag.convertTo(mag_u8, CV_8UC1, 255.0 / sFlowMax, 0);
                cv::applyColorMap(mag_u8, rawFlowVis, cv::COLORMAP_JET);
            }
            else
            {
                rawFlowVis = cv::Mat::zeros(targetSize, CV_8UC3);
            }

            // Push to Viewer under lock
            {
                std::unique_lock<std::mutex> lock(mpViewer->mMutexPAFinsh);
                mpViewer->mDebugRawYoloMask = rawYoloVis.clone();
                mpViewer->mDebugRefinedMask = refinedVis.clone();
                mpViewer->mDebugResidual = residualVis.clone();
                mpViewer->mDebugRefinedMotionMask = refinedMotionVis.clone();
                mpViewer->mDebugRawFlow = rawFlowVis.clone();
                mpViewer->mnDebugFrameId = static_cast<int>(mCurrentFrame.mnId);
                mpViewer->mbDebugDataReady = true;
            }
        }
    }
}

// ------------------------------------
// Original CheckFlowConsistency (Deprecated or kept for reference)
// ------------------------------------
void Tracking::CheckFlowConsistency()
{
    // 0. 安全检查
    // 确保有上一帧图像和当前帧图像用于计算光流，且当前帧有特征点
    if(mLastImGray.empty() || mImGray.empty() || mCurrentFrame.N == 0)
        return;

    // 1. 光流跟踪准备
    // 我们需要把当前帧的所有特征点 (KeyPoints) 分为两部分：
    // Set Q: 背景点 (Background)，即在 YOLO 动态框之外的点。
    // Set D: 嫌疑点 (Dynamic Candidates)，即在 YOLO 动态框之内的点。
    // 我们将计算这所有点的光流，用 Set Q 来估计相机的背景运动，然后用 Set D 来做校验。

    // 获取动态框
    const std::vector<cv::Rect2i>& vDynamicAreas = mpORBextractorLeft->mvDynamicArea;
    if(vDynamicAreas.empty()) 
    {
        // 如果没有动态框，理论上不需要做剔除，或者假设所有点都是背景
        // 但为了严谨，我们直接返回
        return;
    }

    // 1. 数据准备 (Data Preparation)
    // 目标：计算当前帧所有特征点在上一帧的对应位置。
    // 方法：使用 LK 光流法进行反向追踪 (Current -> Previous)。
    
    vector<cv::Point2f> vPtsCurr;
    vPtsCurr.reserve(mCurrentFrame.N);
    for(int i=0; i<mCurrentFrame.N; i++)
    {
        vPtsCurr.push_back(mCurrentFrame.mvKeysUn[i].pt);
    }

    vector<cv::Point2f> vPtsPrev;
    vector<uchar> status;
    vector<float> err;

    // 2. 运行 LK 光流 (Run Optical Flow)
    // 输入：当前帧 (mImGray) 和 上一帧 (mLastImGray)
    // 输出：vPtsCurr 中每个点在上一帧的坐标 vPtsPrev
    cv::calcOpticalFlowPyrLK(mImGray, mLastImGray, vPtsCurr, vPtsPrev, status, err, cv::Size(21,21), 3);

    // 3. 特征点分类 (Set Division) 与 背景光流计算 (Background Flow Calculation)
    // 集合 Q (Static Set): 位于 YOLO 检测框外部的点，被认为是绝对静态背景。
    // 集合 D (Dynamic Set): 位于 YOLO 检测框内部的点，是“嫌疑动态”点。
    
    
    
    // 使用 Median (中位数) 代替 Mean (均值) 以增强鲁棒性，防止背景集混入动态杂点
    vector<float> vFlowU_Q, vFlowV_Q;
    vFlowU_Q.reserve(mCurrentFrame.N);
    vFlowV_Q.reserve(mCurrentFrame.N);

    cout<<"RDFP-SLAM: Debug - CurrentFrame N: "<<mCurrentFrame.N<<endl;

    for(int i=0; i<mCurrentFrame.N; i++)
    {
        if(!status[i]) continue; // 光流追踪失败的点跳过
        
        cv::Point2f pt_curr = vPtsCurr[i];
        cv::Point2f pt_prev = vPtsPrev[i];

        bool bInsideMask = false;
        for(const cv::Rect2i& rect : vDynamicAreas)
        {
            if(rect.contains(pt_curr))
            {
                bInsideMask = true;
                break;
            }
        }

        // 如果在框外，属于集合 Q (背景点)
        if(!bInsideMask) 
        {
            float u = pt_curr.x - pt_prev.x;
            float v = pt_curr.y - pt_prev.y;
            
            vFlowU_Q.push_back(u);
            vFlowV_Q.push_back(v);
        }
    }

    if(vFlowU_Q.size() < 10) 
    {
        std::cout << "RDFP-SLAM: Debug - Too few background points (Q=" << vFlowU_Q.size() << ")" << std::endl;
        return; 
    }

    // 计算背景中位数光流 (Background Median Flow)
    std::sort(vFlowU_Q.begin(), vFlowU_Q.end());
    std::sort(vFlowV_Q.begin(), vFlowV_Q.end());
    
    float median_u = vFlowU_Q[vFlowU_Q.size() / 2];
    float median_v = vFlowV_Q[vFlowV_Q.size() / 2];

    cout<<"RDFP-SLAM: Debug - median_u: "<<median_u<<endl;
    cout<<"RDFP-SLAM: Debug - median_v: "<<median_v<<endl;

    // 4. 动态判定与剔除 (Verification & Rejection)
    // 阈值设定：如果相对速度差大于 3.0 像素，则认为是动态点
    float th_flow = 3.0f; 
    int nOutliers = 0;
    int count_D = 0;
    float max_diff = 0.0f;

    // RDFP-SLAM: Initialize Flow Mask for Dense Filtering
    cv::Mat flowMask = cv::Mat::zeros(mImGray.size(), CV_8UC1);
    
    // Store outliers grouped by Bounding Box index
    // vDynamicAreas has size K. outliersPerBox[k] stores points for k-th box.
    std::vector<std::vector<cv::Point2f>> outliersPerBox(vDynamicAreas.size());

    for(int i=0; i<mCurrentFrame.N; i++)
    {
        if(!status[i]) continue;

        cv::Point2f pt_curr = vPtsCurr[i];
        int boxIdx = -1;
        for(size_t k=0; k<vDynamicAreas.size(); k++)
        {
            if(vDynamicAreas[k].contains(pt_curr))
            {
                boxIdx = k;
                break;
            }
        }

        // 仅对集合 D (框内嫌疑点) 进行校验
        if(boxIdx != -1) 
        {
            count_D++;
            cv::Point2f pt_prev = vPtsPrev[i];
            float u = pt_curr.x - pt_prev.x;
            float v = pt_curr.y - pt_prev.y;

            float diff_u = std::abs(u - median_u);
            float diff_v = std::abs(v - median_v);
            
            float current_diff = std::max(diff_u, diff_v);
            if(current_diff > max_diff) max_diff = current_diff;

            // 剔除逻辑
            if(diff_u > th_flow || diff_v > th_flow)
            {
                // 确实是动态点 (True Dynamic)，进行剔除
                mCurrentFrame.mvbOutlier[i] = true;
                
                // Tag for Viewer: specific flow-based outlier
                mCurrentFrame.mvKeys[i].class_id = -99;
                
                // 同时尝试设置 MapPoint 的坏点标志 (如果有对应的 MapPoint)
                if(mCurrentFrame.mvpMapPoints[i])
                {
                    mCurrentFrame.mvpMapPoints[i]->SetBadFlag();
                    // mCurrentFrame.mvpMapPoints[i] = static_cast<MapPoint*>(NULL); // KEEP ptr for visualization in FrameDrawer
                }
                
                // Collect outlier for Hull computation
                outliersPerBox[boxIdx].push_back(pt_curr);
                
                nOutliers++;
            }
        }
    }

    // **改进策略：基于光流验证的YOLO Mask自适应扩展**
    // 目标：保留YOLO的完整覆盖，用光流验证运动，自适应调整膨胀大小
    
    cv::Mat refinedMask = cv::Mat::zeros(mImGray.size(), CV_8UC1);
    bool bMaskUpdated = false;
    
    // DEBUG: 检查YOLO masks是否可用
    std::cout << "RDFP-SLAM Debug: vDynamicAreas.size()=" << vDynamicAreas.size() 
              << ", mpORBextractorLeft->mvDynamicMask.size()=" << mpORBextractorLeft->mvDynamicMask.size() << std::endl;
    
    for(size_t k=0; k<vDynamicAreas.size(); k++)
    {
        const cv::Rect2i& box = vDynamicAreas[k];
        int nFlowOutliers = outliersPerBox[k].size();
        
        // 获取原始YOLO mask
        cv::Mat yoloMaskForBox;
        bool hasYoloMask = (k < mpORBextractorLeft->mvDynamicMask.size() && 
                           !mpORBextractorLeft->mvDynamicMask[k].empty());
        
        if(hasYoloMask) {
            yoloMaskForBox = mpORBextractorLeft->mvDynamicMask[k].clone();
        } else {
            // YOLO mask缺失，跳过此box（提高性能）
            std::cout << "WARNING: YOLO mask " << k << " missing, skipping this box!" << std::endl;
            continue;
        }
        
        // 根据光流outlier数量决定扩展程度
        int kernelSize;
        const char* motionStatus;
        
        if(nFlowOutliers >= 10) {
            // 高置信度：物体确实在运动，需要大范围扩展覆盖运动模糊
            kernelSize = 61;  // 大膨胀
            motionStatus = "HIGH_MOTION";
            bMaskUpdated = true;
        } 
        else if(nFlowOutliers >= 3) {
            // 中等置信度：部分运动，中等扩展
            kernelSize = 31;
            motionStatus = "MEDIUM_MOTION";
            bMaskUpdated = true;
        }
        else if(nFlowOutliers >= 1) {
            // 低置信度：可能有轻微运动
            kernelSize = 15;
            motionStatus = "LOW_MOTION";  
            bMaskUpdated = true;
        }
        else {
            // 无运动证据：保持YOLO原始结果（可能是误检或静态）
            kernelSize = 6;  // YOLO原始膨胀大小
            motionStatus = "NO_MOTION_DETECTED";
        }
        
        // 应用自适应膨胀到YOLO mask
        cv::Mat kernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(kernelSize, kernelSize));
        cv::Mat expandedMask;
        cv::dilate(yoloMaskForBox, expandedMask, kernel);
        
        // 合并到refined mask
        cv::bitwise_or(refinedMask, expandedMask, refinedMask);
        
        std::cout << "RDFP-SLAM Box[" << k << "]: " << motionStatus 
                  << " (outliers=" << nFlowOutliers << ", kernel=" << kernelSize << ")" << std::endl;
    }
    
    // 用refined mask替换所有YOLO masks
    if(!vDynamicAreas.empty())
    {
        mCurrentFrame.mvDynamicMask.clear();  // 清除原始YOLO masks
        mCurrentFrame.mvDynamicMask.push_back(refinedMask);  // 添加refined mask
        
        std::cout << "RDFP-SLAM Summary: " << vDynamicAreas.size() << " boxes processed, "
                  << nOutliers << " total outliers, max_diff=" << max_diff
                  << ", bg_flow=(" << median_u << "," << median_v << ")" << std::endl;
    }
}

} // namespace ORB_SLAM3
