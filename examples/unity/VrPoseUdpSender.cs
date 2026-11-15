using System;
using System.Collections.Generic;
using System.Net.Sockets;
using System.Text;
using UnityEngine;
using UnityEngine.XR;

// Unity 6 / XR Feature API. OpenXR may use SteamVR as its runtime.
// No Valve plugin is required. This sends poses, not robot joint commands.
public sealed class VrPoseUdpSender : MonoBehaviour
{
    public string receiverHost = "127.0.0.1";
    public int receiverPort = 15010;
    [Range(1f, 200f)] public float sendRateHz = 100f;

    [Serializable]
    private sealed class PosePacket
    {
        public float[] position = { 0f, 0f, 0f };
        public float[] rotation_xyzw = { 0f, 0f, 0f, 1f };
        public bool tracked;
        public bool enabled;
        public long activation_id;
    }

    [Serializable]
    private sealed class ControllerPacket
    {
        public int version = 1;
        public string type = "vr_controller";
        public long sequence;
        public double timestamp;
        public string coordinate_system = "unity";
        public PosePacket left;
        public PosePacket right;
    }

    private UdpClient udp;
    private long sequence;
    private double lastTimestamp, nextSend, nextSubsystemRefresh, nextWarning;
    private bool leftGrip, rightGrip, leftRequireRelease, rightRequireRelease;
    private bool lastLeftTracked, lastRightTracked;
    private long leftActivation, rightActivation;
    private InputDevice leftDevice, rightDevice;
    private bool leftDeviceSeen, rightDeviceSeen;
    private bool hasFocus = true, paused;
    private volatile bool originChanged;
    private volatile bool leftDisconnected, rightDisconnected;
    private readonly List<XRInputSubsystem> scratchSubsystems = new List<XRInputSubsystem>();
    private readonly List<XRInputSubsystem> subscribedSubsystems = new List<XRInputSubsystem>();

    private void OnEnable()
    {
        try
        {
            if (receiverPort < 1 || receiverPort > 65535)
                throw new ArgumentOutOfRangeException("receiverPort");
            udp = new UdpClient(AddressFamily.InterNetwork);
            udp.Client.SendTimeout = 10;
            udp.Connect(receiverHost, receiverPort);
            nextSend = nextSubsystemRefresh = 0;
            InputDevices.deviceDisconnected += OnDeviceDisconnected;
            RefreshSubsystems();
        }
        catch (Exception error)
        {
            Debug.LogError("VR pose sender failed to open: " + error.Message, this);
            CloseSocket();
            enabled = false;
        }
    }

    private void RefreshSubsystems()
    {
        scratchSubsystems.Clear();
        SubsystemManager.GetSubsystems(scratchSubsystems);
        foreach (XRInputSubsystem subsystem in scratchSubsystems)
        {
            if (subscribedSubsystems.Contains(subsystem)) continue;
            subsystem.trackingOriginUpdated += OnTrackingOriginUpdated;
            subscribedSubsystems.Add(subsystem);
        }
    }

    private void OnTrackingOriginUpdated(XRInputSubsystem subsystem)
    {
        // Handle Unity APIs and the stop datagram on the next main-thread update.
        originChanged = true;
    }

    private void OnDeviceDisconnected(InputDevice device)
    {
        if (leftDeviceSeen && device.Equals(leftDevice)) leftDisconnected = true;
        if (rightDeviceSeen && device.Equals(rightDevice)) rightDisconnected = true;
    }

    private void Update()
    {
        if (udp == null) return;
        double now = Time.realtimeSinceStartupAsDouble;
        if (now >= nextSubsystemRefresh || subscribedSubsystems.Count == 0)
        {
            RefreshSubsystems();
            nextSubsystemRefresh = now + 1.0;
        }
        if (originChanged)
        {
            originChanged = false;
            SendStopPacket();
        }
        if (!hasFocus || paused) return;

        bool disconnected = leftDisconnected || rightDisconnected;
        if (leftDisconnected) { leftRequireRelease = true; leftDisconnected = false; }
        if (rightDisconnected) { rightRequireRelease = true; rightDisconnected = false; }

        bool previousLeftGrip = leftGrip, previousRightGrip = rightGrip;
        PosePacket left = ReadPose(XRNode.LeftHand, ref leftGrip, ref leftActivation, ref leftRequireRelease,
                                   ref leftDevice, ref leftDeviceSeen);
        PosePacket right = ReadPose(XRNode.RightHand, ref rightGrip, ref rightActivation, ref rightRequireRelease,
                                    ref rightDevice, ref rightDeviceSeen);
        bool edge = disconnected || previousLeftGrip != leftGrip || previousRightGrip != rightGrip ||
                    (lastLeftTracked && !left.tracked) || (lastRightTracked && !right.tracked);
        lastLeftTracked = left.tracked;
        lastRightTracked = right.tracked;
        if (now < nextSend && !edge) return;

        // At most one current sample per Update; never catch up with bursts.
        nextSend = now + 1.0 / Math.Max(1.0, sendRateHz);
        Send(left, right);
    }

    private static bool Finite(float value)
    {
        return !float.IsNaN(value) && !float.IsInfinity(value);
    }

    private static PosePacket ReadPose(XRNode node, ref bool knownGrip, ref long activation,
                                       ref bool requireRelease, ref InputDevice previousDevice, ref bool deviceSeen)
    {
        InputDevice device = InputDevices.GetDeviceAtXRNode(node);
        if (device.isValid)
        {
            // InputDevice.Equals compares its internal device ID. That ID is
            // private in Unity's Feature API, so do not access it directly.
            if (deviceSeen && !device.Equals(previousDevice)) requireRelease = true;
            previousDevice = device;
            deviceSeen = true;
        }
        bool grip = false, isTracked = false;
        Vector3 position = Vector3.zero;
        Quaternion rotation = Quaternion.identity;
        InputTrackingState tracking = InputTrackingState.None;
        bool haveGrip = device.isValid && device.TryGetFeatureValue(CommonUsages.gripButton, out grip);
        bool haveTracked = device.isValid && device.TryGetFeatureValue(CommonUsages.isTracked, out isTracked);
        bool havePosition = device.isValid && device.TryGetFeatureValue(CommonUsages.devicePosition, out position);
        bool haveRotation = device.isValid && device.TryGetFeatureValue(CommonUsages.deviceRotation, out rotation);
        bool haveTracking = device.isValid && device.TryGetFeatureValue(CommonUsages.trackingState, out tracking);
        InputTrackingState required = InputTrackingState.Position | InputTrackingState.Rotation;
        float norm = Mathf.Sqrt(rotation.x * rotation.x + rotation.y * rotation.y +
                                rotation.z * rotation.z + rotation.w * rotation.w);
        bool poseValid = haveTracked && isTracked && haveTracking && (tracking & required) == required &&
                         havePosition && haveRotation && Finite(position.x) && Finite(position.y) &&
                         Finite(position.z) && Finite(norm) && norm > 0.000001f;
        if (haveGrip)
        {
            if (grip && !knownGrip) activation++;
            knownGrip = grip;
        }
        // Missing input is not evidence that a held grip was released.
        if (!poseValid || !haveGrip) requireRelease = true;
        if (poseValid && haveGrip && !grip) requireRelease = false;
        PosePacket result = new PosePacket {
            tracked = poseValid && haveGrip && !requireRelease,
            enabled = knownGrip,
            activation_id = activation
        };
        if (poseValid)
        {
            result.position = new float[] { position.x, position.y, position.z };
            result.rotation_xyzw = new float[] { rotation.x / norm, rotation.y / norm,
                                                  rotation.z / norm, rotation.w / norm };
        }
        return result;
    }

    private void Send(PosePacket left, PosePacket right)
    {
        if (udp == null) return;
        // Keep release/stop messages strictly ordered even within one render frame.
        lastTimestamp = Math.Max(Time.realtimeSinceStartupAsDouble, lastTimestamp + 0.000001);
        ControllerPacket packet = new ControllerPacket {
            sequence = ++sequence, timestamp = lastTimestamp, left = left, right = right
        };
        byte[] bytes = Encoding.UTF8.GetBytes(JsonUtility.ToJson(packet));
        try { udp.Send(bytes, bytes.Length); }
        catch (Exception error)
        {
            if (lastTimestamp >= nextWarning)
            {
                Debug.LogWarning("VR pose UDP send failed: " + error.Message, this);
                nextWarning = lastTimestamp + 1.0;
            }
        }
    }

    private void SendStopPacket()
    {
        leftRequireRelease = rightRequireRelease = true;
        lastLeftTracked = lastRightTracked = false;
        Send(new PosePacket { tracked = false, enabled = false, activation_id = leftActivation },
             new PosePacket { tracked = false, enabled = false, activation_id = rightActivation });
    }

    private void OnApplicationFocus(bool focus)
    {
        hasFocus = focus;
        if (!focus) SendStopPacket();
        else nextSend = 0;
    }

    private void OnApplicationPause(bool pause)
    {
        paused = pause;
        if (pause) SendStopPacket();
        else nextSend = 0;
    }

    private void OnDisable()
    {
        SendStopPacket();
        InputDevices.deviceDisconnected -= OnDeviceDisconnected;
        foreach (XRInputSubsystem subsystem in subscribedSubsystems)
            subsystem.trackingOriginUpdated -= OnTrackingOriginUpdated;
        subscribedSubsystems.Clear();
        CloseSocket();
    }

    private void OnDestroy() { CloseSocket(); }
    private void CloseSocket()
    {
        if (udp != null) udp.Close();
        udp = null;
    }
}
