// Copyright 2018-present Network Optix, Inc. Licensed under MPL 2.0: www.mozilla.org/MPL/2.0/

#include "device_agent.h"

#include <chrono>
#include <exception>
#include <cctype>

#include <opencv2/core.hpp>
#include <opencv2/dnn/dnn.hpp>

#include <nx/sdk/analytics/helpers/event_metadata.h>
#include <nx/sdk/analytics/helpers/event_metadata_packet.h>
#include <nx/sdk/analytics/helpers/object_metadata.h>
#include <nx/sdk/analytics/helpers/object_metadata_packet.h>
#include <nx/sdk/helpers/string.h>

#include "detection.h"
#include "exceptions.h"
#include "frame.h"

//#include <opencv2/imgproc.hpp>
//#include <opencv2/highgui/highgui.hpp>

//namespace nx {
//namespace vms_server_plugins {
//namespace analytics {
//namespace sample {

namespace sample {
    namespace analytics {
        namespace opencv_object_detection {

            using namespace nx::sdk;
            using namespace nx::sdk::analytics;
            using namespace std::string_literals;

            /**
             * @param deviceInfo Various information about the related device, such as its id, vendor, model,
             *     etc.
             */
            DeviceAgent::DeviceAgent(const nx::sdk::IDeviceInfo* deviceInfo,
                                     std::filesystem::path pluginHomeDir, Ort::Session* core_net):
                // Call the DeviceAgent helper class constructor telling it to verbosely report to stderr.
                ConsumingDeviceAgent(deviceInfo, /*enableOutput*/ true),
                //m_objectDetector(std::make_unique<ObjectDetector>(pluginHomeDir)),
                m_objectDetector(std::make_unique<ObjectDetector>(core_net)),
                m_objectTracker(std::make_unique<ObjectTracker>())
            {
            }

            DeviceAgent::~DeviceAgent()
            {
            }

            /**
             *  @return JSON with the particular structure. Note that it is possible to fill in the values
             * that are not known at compile time, but should not depend on the DeviceAgent settings.
             */
            std::string DeviceAgent::manifestString() const
            {
                // Tell the Server that the plugin can generate the events and objects of certain types.
                // Id values are strings and should be unique. Format of ids:
                // `{vendor_id}.{plugin_id}.{event_type_id/object_type_id}`.
                //
                // See the plugin manifest for the explanation of vendor_id and plugin_id.
                return /*suppress newline*/ 1 + (const char*) R"json(
            {
                "eventTypes": [
                    {
                        "id": ")json" + kDetectionEventType + R"json(",
                        "name": "Object detected"
                    },
                    {
                        "id": ")json" + kProlongedDetectionEventType + R"json(",
                        "name": "Object detected (prolonged)",
                        "flags": "stateDependent"
                    }
                ],
                "objectTypes": [
                    {
                        "id": ")json" + kHumanFallObjectType + R"json(",
                        "name": "Human Fall"
                    },
                    {
                        "id": ")json" + kNonHelmetObjectType + R"json(",
                        "name": "None Helmet"
                    },
                    {
                        "id": ")json" + kFireObjectType + R"json(",
                        "name": "Fire"
                    },
                    {
                        "id": ")json" + kSmokeObjectType + R"json(",
                        "name": "Smoke"
                    },
                    {
                        "id": ")json" + kHeadtObjectType + R"json(",
                        "name": "Head"
                    },
                    {
                        "id": ")json" + kHelmetObjectType + R"json(",
                        "name": "Helmet"
                    }
,
                    {
                        "id": ")json" + kHumanObjectType + R"json(",
                        "name": "Human"
                    },
                    {
                        "id": ")json" + kHalfFallObjectType + R"json(",
                        "name": "Half Fall"
                    }
                ]
            }
            )json";
            }


            /**
             * Called when the Server sends a new uncompressed frame from a camera.
             */
            bool DeviceAgent::pushUncompressedVideoFrame(const IUncompressedVideoFrame* videoFrame)
            {


                m_terminated = m_terminated || m_objectDetector->isTerminated();
                if (m_terminated)
                {
                    if (!m_terminatedPrevious)
                    {
                        pushPluginDiagnosticEvent(
                            IPluginDiagnosticEvent::Level::error,
                            "Plugin is in broken state.",
                            "Disable the plugin.");
                        m_terminatedPrevious = true;
                    }
                    return true;
                }

                
                //m_lastVideoFrameTimestampUs = videoFrame->timestampUs();

                //auto eventMetadataPacket = generateEventMetadataPacket();
                //if (eventMetadataPacket)
                //{
                //    // Send generated metadata packet to the Server.
                //    pushMetadataPacket(eventMetadataPacket.releasePtr());
                //}

                // Detecting objects only on every `kDetectionFramePeriod` frame.
                if (m_frameIndex % kDetectionFramePeriod == 0)
                {
                    const MetadataPacketList metadataPackets = processFrame(videoFrame);
                    for (const Ptr<IMetadataPacket>& metadataPacket : metadataPackets)
                    {
                        metadataPacket->addRef();
                        pushMetadataPacket(metadataPacket.get());
                    }
                }

                ++m_frameIndex;

                return true; //< There were no errors while processing the video frame.
            }

            /**
             * Serves the similar purpose as pushMetadataPacket(). The differences are:
             * - pushMetadataPacket() is called by the plugin, while pullMetadataPackets() is called by Server.
             * - pushMetadataPacket() expects one metadata packet, while pullMetadataPacket expects the
             *     std::vector of them.
             *
             * There are no strict rules for deciding which method is "better". A rule of thumb is to use
             * pushMetadataPacket() when you generate one metadata packet and do not want to store it in the
             * class field, and use pullMetadataPackets otherwise.
             */

            //bool DeviceAgent::pullMetadataPackets(std::vector<IMetadataPacket*>* metadataPackets)
            //{
            //    metadataPackets->push_back(generateObjectMetadataPacket().releasePtr());

            //    return true; //< There were no errors while filling metadataPackets.
            //}

            void DeviceAgent::doSetNeededMetadataTypes(
                //nx::sdk::Result<void>* /*outValue*/,
                nx::sdk::Result<void>* outValue,
                const nx::sdk::analytics::IMetadataTypes* /*neededMetadataTypes*/)
            {
                if (m_terminated)
                    return;

                try
                {
                    m_objectDetector->ensureInitialized();  // Load Model
                }
                catch (const ObjectDetectorInitializationError& e)
                {
                    *outValue = { ErrorCode::otherError, new String(e.what()) };
                    m_terminated = true;
                }
                catch (const ObjectDetectorIsTerminatedError& /*e*/)
                {
                    m_terminated = true;
                }
            }

            //-------------------------------------------------------------------------------------------------
            // private
            DeviceAgent::MetadataPacketList DeviceAgent::eventsToEventMetadataPacketList(
                const EventList& events, 
                int64_t timestampUs)
            {
                if (events.empty())
                    return {};

                MetadataPacketList result;

                const auto objectDetectedEventMetadataPacket = makePtr<EventMetadataPacket>();
                
                ////single event (1)
                //const auto eventMetadata = makePtr<EventMetadata>();
                //eventMetadata->setCaption(caption);
                //eventMetadata->setDescription(description);
                //eventMetadata->setIsActive(event->eventType == EventType::detection_started);
                //eventMetadata->setTypeId(kProlongedDetectionEventType);

                //const auto eventMetadataPacket = makePtr<EventMetadataPacket>();
                //eventMetadataPacket->addItem(eventMetadata.get());
                //eventMetadataPacket->setTimestampUs(event->timestampUs);
                //result.push_back(eventMetadataPacket);

                //objectDetectedEventMetadataPacket->setTimestampUs(timestampUs);
                //result.push_back(objectDetectedEventMetadataPacket);

                ////single detectionList (1)
                //const auto eventMetadata = makePtr<EventMetadata>();
                //eventMetadata->setCaption(caption);
                //eventMetadata->setDescription(description);
                //eventMetadata->setIsActive(event->eventType == EventType::detection_started);
                //eventMetadata->setTypeId(kProlongedDetectionEventType);

                //const auto eventMetadataPacket = makePtr<EventMetadataPacket>();
                //eventMetadataPacket->addItem(eventMetadata.get());
                //eventMetadataPacket->setTimestampUs(event->timestampUs);
                //result.push_back(eventMetadataPacket);

                //objectDetectedEventMetadataPacket->setTimestampUs(timestampUs);
                //result.push_back(objectDetectedEventMetadataPacket);

                for (const std::shared_ptr<Event>& event : events)
                {
                    const auto eventMetadata = makePtr<EventMetadata>();

                    if (event->eventType == EventType::detection_started ||
                        event->eventType == EventType::detection_finished)
                    {
                        static const std::string kStartedSuffix = " STARTED";
                        static const std::string kFinishedSuffix = " FINISHED";

                        const std::string suffix = (event->eventType == EventType::detection_started) ?
                            kStartedSuffix : kFinishedSuffix;
                        //const std::string caption = kClassesToDetectPluralCapitalized.at(event->classLabel) +
                        //    " detection" + suffix;
                        const std::string caption = (event->classLabel) +
                            " detection" + suffix;
                        const std::string description = caption;

                        eventMetadata->setCaption(caption);
                        eventMetadata->setDescription(description);
                        eventMetadata->setIsActive(event->eventType == EventType::detection_started);
                        eventMetadata->setTypeId(kProlongedDetectionEventType);

                        const auto eventMetadataPacket = makePtr<EventMetadataPacket>();
                        eventMetadataPacket->addItem(eventMetadata.get());
                        eventMetadataPacket->setTimestampUs(event->timestampUs);
                        result.push_back(eventMetadataPacket);
                    }
                    else if (event->eventType == EventType::object_detected)
                    {
                        std::string caption = event->classLabel + kDetectionEventCaptionSuffix;
                        caption[0] = (char)toupper(caption[0]);
                        std::string description = event->classLabel + kDetectionEventDescriptionSuffix;
                        description[0] = (char)toupper(description[0]);

                        eventMetadata->setCaption(caption);
                        eventMetadata->setDescription(description);
                        eventMetadata->setIsActive(true);
                        eventMetadata->setTypeId(kDetectionEventType);

                        objectDetectedEventMetadataPacket->addItem(eventMetadata.get());
                    }
                }

                objectDetectedEventMetadataPacket->setTimestampUs(timestampUs);
                objectDetectedEventMetadataPacket->setDurationUs(200000); //generate event for 0.2 seconds

                result.push_back(objectDetectedEventMetadataPacket);

                return result;
            }

            Ptr<ObjectMetadataPacket> DeviceAgent::detectionsToObjectMetadataPacket(
                const DetectionList& detections,
                int64_t timestampUs)
            {
                if (detections.empty())
                    return nullptr;

                const auto objectMetadataPacket = makePtr<ObjectMetadataPacket>();

                for (const std::shared_ptr<Detection>& detection : detections)
                {
                    const auto objectMetadata = makePtr<ObjectMetadata>();

                    objectMetadata->setBoundingBox(detection->boundingBox);
                    objectMetadata->setConfidence(detection->confidence);
                    objectMetadata->setTrackId(detection->trackId);

                    // Convert class label to object metadata type id.
                    if (detection->classLabel == "human_fall")
                        objectMetadata->setTypeId(kHumanFallObjectType);
                    else if (detection->classLabel == "non_helmet")
                        objectMetadata->setTypeId(kNonHelmetObjectType);
                    else if (detection->classLabel == "fire")
                        objectMetadata->setTypeId(kFireObjectType);
                    else if (detection->classLabel == "smoke")
                        objectMetadata->setTypeId(kSmokeObjectType);
                    else if (detection->classLabel == "head")
                        objectMetadata->setTypeId(kHeadtObjectType);
                    else if (detection->classLabel == "helmet")
                        objectMetadata->setTypeId(kHelmetObjectType);
                    else if (detection->classLabel == "human")
                        objectMetadata->setTypeId(kHumanObjectType);
                    else if (detection->classLabel == "half-fall")
                        objectMetadata->setTypeId(kHalfFallObjectType);

                    // There is no "else", because only the detections with those types are generated.

                    objectMetadataPacket->addItem(objectMetadata.get());
                }



                objectMetadataPacket->setTimestampUs(timestampUs);
                objectMetadataPacket->setDurationUs(200000); // draw bounding box for 0.2 seconds

                return objectMetadataPacket;
            }


            void DeviceAgent::reinitializeObjectTrackerOnFrameSizeChanges(const Frame& frame)
            {
                const bool frameSizeUnset = m_previousFrameWidth == 0 && m_previousFrameHeight == 0;
                if (frameSizeUnset)
                {
                    m_previousFrameWidth = frame.width;
                    m_previousFrameHeight = frame.height;
                    return;
                }

                const bool frameSizeChanged = frame.width != m_previousFrameWidth ||
                    frame.height != m_previousFrameHeight;
                if (frameSizeChanged)
                {
                    m_objectTracker = std::make_unique<ObjectTracker>();
                    m_previousFrameWidth = frame.width;
                    m_previousFrameHeight = frame.height;
                }
            }


            DeviceAgent::MetadataPacketList DeviceAgent::processFrame(
                const IUncompressedVideoFrame* videoFrame)
            {
                const Frame frame(videoFrame, m_frameIndex);

              /*  cv::imwrite("C:/test/" + std::to_string(imgCount)+".PNG", frame.cvMat);
                imgCount++;*/

                reinitializeObjectTrackerOnFrameSizeChanges(frame);

                //if (!frame.cvMat.empty()) {
                //    cv::Mat image = frame.cvMat;
                //    cv::imwrite("C:/test_compress.png", image);
                //}
                if (frame.cvMat.empty()) {
                    pushPluginDiagnosticEvent(
                        IPluginDiagnosticEvent::Level::warning,
                        "No frame.",
                        "Disable the plugin.");
                }

                try
                {
                    DetectionList detections = m_objectDetector->run(frame);


                    ObjectTracker::Result objectTrackerResult = m_objectTracker->run(frame, detections);


                    const auto& objectMetadataPacket =
                        detectionsToObjectMetadataPacket(detections, frame.timestampUs);

                    const auto& eventMetadataPacketList = eventsToEventMetadataPacketList(
                        objectTrackerResult.events,
                        frame.timestampUs);

                    MetadataPacketList result;

                    if (objectMetadataPacket)
                        result.push_back(objectMetadataPacket);


                    result.insert(
                        result.end(),
                        std::make_move_iterator(eventMetadataPacketList.begin()),
                        std::make_move_iterator(eventMetadataPacketList.end()));

                    return result;
                }
                catch (const ObjectDetectionError& e)
                {
                    pushPluginDiagnosticEvent(
                        IPluginDiagnosticEvent::Level::error,
                        "Object detection error.",
                        e.what());
                    m_terminated = true;
                }

                catch (const ObjectTrackingError& e)
                {
                    pushPluginDiagnosticEvent(
                        IPluginDiagnosticEvent::Level::error,
                        "Object tracking error.",
                        e.what());
                    m_terminated = true;
                }

                return {};
            }


        } // namespace opencv_object_detection
    } // namespace analytics
} // namespace sample

//} // namespace sample
//} // namespace analytics
//} // namespace vms_server_plugins
//} // namespace nx
