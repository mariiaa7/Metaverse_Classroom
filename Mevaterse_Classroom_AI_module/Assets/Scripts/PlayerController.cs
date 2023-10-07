using UnityEngine;
using UnityEngine.InputSystem;
using Photon.Pun;
using TMPro;
using UnityEngine.EventSystems;
using Photon.Realtime;
using System;
using System.Threading;
using UnityEngine.Networking;
using System.Text;
using System.Collections;
using System.Diagnostics;



public class PlayerController : MonoBehaviourPunCallbacks
{
    // Controls the camera movement
    [Header("Camera")]
    public Transform playerRoot;
    public Transform playerCam;
    public float cameraSensitivity;
    private float rotX;
    private float rotY;

    [Header("Movement")]
    public CharacterController controller;
    public float speed;
    public float gravity;
    public Transform feet;
    public bool isGrounded;
    Vector3 velocity;

    [Header("Input")]
    public InputAction move;
    public InputAction mouseX;
    public InputAction mouseY;

    // Controls forward and backward movement speed
    private float originalSpeed;
    private float backwardSpeed;

    // Variables for animation control
    private bool isMoving;
    private bool isBackwardMoving;
    private bool isClapping;
    private bool handRaised;
    private bool isWaving;
    public bool isTyping;
    private TextChat textChat;
    public TMP_Text volumeIcon;
    public TMP_Text playerName;
    public TMP_Text emotion;
    public Transform overhead;
    private string boardText;
    private float idleTime;
    private float handRaiseCooldown;

    // Variables for the sitting control
    private GameObject chair;
    private WhiteBoard whiteBoard;
    private bool isSitting;
    private Vector3 originalPosition;
    private float originalFov;

    public Animator animatorController;
    private TMP_Text interactionInfo;
    private ControlInfoHanlder commandInfo;
    private Vector3 spawnPosition;
    public AudioSource clapSound;

    //IA variables
    string pName; 
    public Process ps = new Process();
    public int emotionIndex;
    public EmotionScript script;
    private string[] emotions = { "Angry", "Disgust", "Fear", "Happy", "Neutral", "Sad", "Surprise" };

    

    public override void OnEnable()
    {
        move.Enable();
        mouseX.Enable();
        mouseY.Enable();
        textChat = GameObject.Find("TextChat").GetComponent<TextChat>();
    }

    public override void OnDisable()
    {
        move.Disable();
        mouseX.Disable();
        mouseY.Disable();
    }

    private void Start()
    {
        script = GameObject.FindObjectOfType<EmotionScript>();
        ps.StartInfo.FileName = "cmd.exe";
        //ps.StartInfo.WindowStyle = ProcessWindowStyle.Hidden;
        ps.StartInfo.Arguments = @"/c uvicorn Assets.Scripts.model_fer:app --reload";
        ps.Start();


        StartCoroutine(GetRequest("http://127.0.0.1:8000/"));

        playerCam = GameObject.FindWithTag("MainCamera").transform;
        playerName.text = GetComponent<PhotonView>().Controller.NickName;
        pName = playerName.text;

        volumeIcon.text = ""; 
        boardText = $"{DateTime.UtcNow.Date.ToString("MM/dd/yyyy")}";
        whiteBoard = null;
        commandInfo = GameObject.Find("CommandInfo").GetComponent<ControlInfoHanlder>();


        photonView.RPC("NotifySpawnRPC", RpcTarget.All);
        GameObject.Find("WelcomeAudioSource").GetComponent<AudioSource>().Play();
        GameObject.Find("BgAudioSource").GetComponent<AudioSource>().enabled = false;


        Cursor.lockState = CursorLockMode.Locked;

        // Variables inizialitazion
        controller = GetComponent<CharacterController>();
        idleTime = 0;
        handRaiseCooldown = 10;
        originalSpeed = speed;
        originalFov = playerCam.GetComponent<Camera>().fieldOfView;
        backwardSpeed = originalSpeed - 1.3f;
        handRaised = false;
        isSitting = false;
        isClapping = false;

        interactionInfo = GameObject.Find("InteractionInfo").GetComponent<TMP_Text>();
        interactionInfo.text = "";
        spawnPosition = GameObject.Find("SpawnPosition").GetComponent<Transform>().position;

        gameObject.SetActive(false);
        transform.position = spawnPosition + new Vector3((float)(-photonView.ViewID)/10000f, 0, 0);        
        gameObject.SetActive(true);
    }

    
    IEnumerator GetRequest(string uri)
    {
        using(UnityWebRequest webRequest = UnityWebRequest.Get(uri))
        {
            yield return webRequest.SendWebRequest();
            switch(webRequest.result)
            {
                case UnityWebRequest.Result.ConnectionError:
                case UnityWebRequest.Result.DataProcessingError:
                    //UnityEngine.Debug.LogError(string.Format("Error: {0}", webRequest.error));
                    break;
                case UnityWebRequest.Result.Success:
                    emotionIndex = int.Parse(webRequest.downloadHandler.text);
                    UnityEngine.Debug.Log(emotionIndex);
                    if(emotionIndex >= 0 && emotionIndex <= 6)
                    switch(emotionIndex){
                        case 0: script.angry.text = "Angry: " + ++script.angryCount; break;
                        case 1: script.disgust.text = "Disgust: " + ++script.disgustCount; break;
                        case 2: script.fear.text = "Fear: " + ++script.fearCount; break;
                        case 3: script.happy.text = "Happy: " + ++script.happyCount; break;
                        case 4: script.neutral.text = "Neutral: " + ++script.neutralCount; break;
                        case 5: script.sad.text = "Sad: " + ++script.sadCount; break;
                        case 6: script.surprise.text = "Surprise: " + ++script.surpriseCount; break;
                        default: break;
                    }
                    break;
            }
        }
    }

    private void Update()
    {
      

        if (!photonView.IsMine) { return; }

        controller.Move(velocity * Time.deltaTime);

        // Camera Movement
        Vector2 mouseInput = new Vector2(mouseX.ReadValue<float>() * cameraSensitivity, mouseY.ReadValue<float>() * cameraSensitivity);
        rotX -= mouseInput.y;
        rotX = Mathf.Clamp(rotX, -30, +50);

        if(!isSitting)
            rotY += mouseInput.x;

        else if(isSitting & !isTyping)
        {
            rotY += mouseInput.x;
            rotY = Mathf.Clamp(rotY, -90, +90);
        }

        else if(isSitting & isTyping)
        {
            rotY += mouseInput.x;
            rotY = Mathf.Clamp(rotY, -10, +10);
        }

        playerRoot.rotation = Quaternion.Euler(0f, rotY, 0f);
        //playerCam.localRotation = Quaternion.Euler(rotX, 0f, 0f);
        playerCam.rotation = Quaternion.Euler(rotX, 0f, 0f);


        // Player Movement
        Vector2 moveInput = move.ReadValue<Vector2>();
        Vector3 moveVelocity = playerRoot.forward * moveInput.y + playerRoot.right * moveInput.x;        
        
        controller.Move(moveVelocity * speed * Time.deltaTime);

        isGrounded = Physics.Raycast(feet.position, feet.TransformDirection(Vector3.down), 0.50f);

        if (isGrounded)
        {
            velocity = new Vector3(0f, -3f, 0f);
        }
        else
        {
            velocity -= gravity * Time.deltaTime * Vector3.up;
        }

        // Player sitting 
        if (Input.GetKeyUp(KeyCode.C) && chair != null && !chair.GetComponent<ChairController>().IsBusy() && !isSitting && !isMoving && !isBackwardMoving && !textChat.isSelected)
        {
            Seat();
        }

        else if (Input.GetKeyUp(KeyCode.C) && isSitting && !Input.GetKey(KeyCode.W) && 
            !Input.GetKey(KeyCode.A) && !Input.GetKey(KeyCode.S) && !Input.GetKey(KeyCode.D) && !textChat.isSelected && !isTyping)
        {
            GetUp();
        }

        // Player writing on whiteboard
        if (Input.GetKeyUp(KeyCode.Space) && whiteBoard != null && !whiteBoard.isBeingEdited && !textChat.isSelected)
        {
            EditWhiteboard();
        }

        else if (Input.GetKeyUp(KeyCode.Escape) && whiteBoard != null && whiteBoard.isBeingEdited && 
            Presenter.Instance.writerID == PhotonNetwork.LocalPlayer.UserId && !textChat.isSelected)
        {
            StopEditWhiteboard();
        }

        if (handRaiseCooldown > 0)
            handRaiseCooldown -= Time.deltaTime;

        // If the player presses M, the character raises their hand
        if (Input.GetKeyUp(KeyCode.M) && handRaiseCooldown <= 0 && !textChat.isSelected && !isTyping)
        {
            RaiseHand();
            handRaiseCooldown = 10;
        }

        if (textChat.isSelected || isTyping)
        {
            controller.enabled = false;
        }
        else if(!textChat.isSelected && !isSitting && !isTyping && controller.enabled == false)
        {
            controller.enabled = true;
        }

        AnimatorChecker(moveVelocity);
        InteractionInfoUpdate();

    }

    private void FixedUpdate()
    {
        StartCoroutine(GetRequest("http://127.0.0.1:8000/"));
    }

    private void LateUpdate()
    {
        // Locks and unlocks the mouse if the player press ESC or the right mouse button
        if (Input.GetKeyUp(KeyCode.Escape))
            Cursor.lockState = CursorLockMode.None;

        if ((Cursor.lockState == CursorLockMode.None) && Input.GetMouseButton(1))
            Cursor.lockState = CursorLockMode.Locked;

        if (Input.GetKey(KeyCode.LeftControl) && Input.GetKeyUp(KeyCode.L))
        {
            string msg = "";
            Logger.Instance.LogInfo($"Player in the room: {PhotonNetwork.PlayerList.Length}");
            foreach (Player p in PhotonNetwork.PlayerList)
                msg += " " + p.NickName;
            Logger.Instance.LogInfo($"Players: {msg}");
        }

            GetComponent<AudioSource>().enabled = (isMoving || isBackwardMoving) && !isSitting;
    }

    void AnimatorChecker(Vector3 moveVelocity)
    {
        isMoving = ((moveVelocity.x != 0 || moveVelocity.y != 0 || moveVelocity.z != 0) && !textChat.isSelected && !isTyping);
        isBackwardMoving = false;
        handRaised = false;
        isWaving = false;
        isClapping = false;
        isTyping = /*(GetComponent<TabletSpawner>().tablet.GetComponent<TabletManager>().isBeingEdited) || */(PhotonNetwork.LocalPlayer.UserId == Presenter.Instance.writerID);

        // If the player is walking backward, this changes the animation and slows down the speed
        if (Input.GetKey(KeyCode.S) && !textChat.isSelected && !isTyping)
        {
            isBackwardMoving = true;
            isMoving = false;
            animatorController.SetBool("IsMovingBackward", true);
            speed = backwardSpeed;

        }

        // When the backward walking is done, it brings the original values back
        else
        {
            isBackwardMoving = false;
            speed = originalSpeed;
            animatorController.SetBool("IsMovingBackward", false);
        }

        if (Input.GetKey(KeyCode.M) && handRaiseCooldown <= 0 && !textChat.isSelected && !isTyping)
        {
            handRaised = true;
        }

        if (Input.GetKey(KeyCode.N) && !textChat.isSelected && !isTyping)
        {
            isWaving = true;
        }

        if (Input.GetKey(KeyCode.V) && !textChat.isSelected && !isTyping)
        {
            isClapping = true;
        }

        if (isTyping)
        {
            textChat.inputField.enabled = false;
        }

        else if(!isTyping)
        {
            textChat.inputField.enabled = true;
        }

        // If the player doesn't move for 6 seconds, perform an idle animation
        idleTime += Time.deltaTime;
        if (isMoving || isBackwardMoving)
        {
            idleTime = 0;
            animatorController.SetBool("LongPause", false);
        }

        animatorController.SetBool("LongPause", idleTime >= 30);

        if (idleTime >= 30)
        {
            idleTime = 0;
        }

        animatorController.SetBool("IsMoving", isMoving);        
        animatorController.SetBool("HandRaised", handRaised);
        animatorController.SetBool("IsWaving", isWaving);
        animatorController.SetBool("IsClapping", isClapping);
        animatorController.SetBool("IsTalking", GetComponent<PlayerVoiceController>().isTalking);
        //animatorController.SetBool("IsWriting", isTyping);

        if (photonView.GetComponent<PlayerVoiceController>().isTalking)
            photonView.RPC("NotifyTalkRPC", RpcTarget.All, "<sprite index=0>");
        else photonView.RPC("NotifyTalkRPC", RpcTarget.All, "");

        photonView.RPC("ClapRPC", RpcTarget.All, isClapping);
    }

    private void Seat()
    {
        GameObject.Find("SitAudioSource").GetComponent<AudioSource>().Play();

        // The chair is set to busy and the player who is occupying it is saved
        GetComponent<PhotonView>().RPC("NotifySitting", RpcTarget.All, true, GetComponent<PhotonView>().Controller.NickName);

        // Saves original player position for when they get up
        originalPosition = transform.position;

        // Makes the player position move on the chair
        gameObject.SetActive(false);
        transform.position = chair.transform.position + new Vector3(0, +0.65f, +0.1f);
        playerCam.GetComponent<Camera>().fieldOfView -= 10f;
        GetComponent<CharacterController>().enabled = false;
        //overhead.position += new Vector3(0, -0.5f, 0);
        gameObject.SetActive(true);

        // Starts the sitting animation for the player
        isSitting = true;
        animatorController.SetBool("IsSitting", true);

        //Tablet spawn
        //GetComponent<TabletSpawner>().SetTabletActive(true, transform.position + new Vector3(-0.05f, 0, 0.5f));
    }

    private void GetUp()
    {
        GameObject.Find("SitAudioSource").GetComponent<AudioSource>().Play();

        // The chair is set to free and the player who was occupying it is deleted
        GetComponent<PhotonView>().RPC("NotifySitting", RpcTarget.All, false, "");

        // Makes the player position the original one
        gameObject.SetActive(false);
        transform.position = originalPosition;
        playerCam.GetComponent<Camera>().fieldOfView = originalFov;
        GetComponent<CharacterController>().enabled = true;
        //overhead.position += new Vector3(0, 0.5f, 0);
        gameObject.SetActive(true);

        // Stops the sitting animation for the player
        isSitting = false;
        animatorController.SetBool("IsSitting", false);
        chair = null;

        //Tablet despawn
        //GetComponent<TabletSpawner>().SetTabletActive(false, transform.position);

    }

    private void EditWhiteboard()
    {
        whiteBoard.boardText.readOnly = false; 
        EventSystem.current.SetSelectedGameObject(whiteBoard.gameObject);
        Cursor.lockState = CursorLockMode.None;
        whiteBoard.boardText.caretPosition = whiteBoard.boardText.text.Length;
        GetComponent<CharacterController>().enabled = false;
        commandInfo.enabled = false;
        photonView.RPC("LockBoard", RpcTarget.All, true, PhotonNetwork.LocalPlayer.UserId, "");
    }

    private void StopEditWhiteboard()
    {
        whiteBoard.boardText.readOnly = true;
        EventSystem.current.SetSelectedGameObject(null);
        Cursor.lockState = CursorLockMode.Locked;
        GetComponent<CharacterController>().enabled = true;
        boardText = whiteBoard.boardText.text;
        commandInfo.enabled = true;
        photonView.RPC("LockBoard", RpcTarget.All, false, "none", boardText);
    }
    private void RaiseHand()
    {
        GetComponent<PhotonView>().RPC("NotifyHandRaisedRPC", RpcTarget.All);
    }

    // Checks if the player is near to a chair to sit on
    private void OnCollisionEnter(Collision collision)
    {
        if (collision.gameObject.CompareTag("Chair"))
        {
            chair = collision.gameObject;
        }
    }

    private void OnCollisionExit(Collision collision)
    {
        if (collision.gameObject.CompareTag("Chair"))
        {
            chair = null;
        }
    }

    private void OnTriggerEnter(Collider collision)
    {
        if (collision.gameObject.CompareTag("Whiteboard"))
        {
            whiteBoard = collision.gameObject.GetComponent<WhiteBoard>();
        }
    }
    private void OnTriggerExit(Collider collision)
    {
        if (collision.gameObject.CompareTag("Whiteboard"))
        {
            whiteBoard = null;
        }
    }

    private void InteractionInfoUpdate()
    {
        if (chair != null && !isSitting && !chair.GetComponent<ChairController>().IsBusy())
            interactionInfo.text = "Press C to sit";

        else if (isSitting && !isTyping)
            interactionInfo.text = "Press C to stand up";

        else if(isSitting && isTyping)
            interactionInfo.text = "Press ESC to stop writing";

        else if (chair != null && !isSitting && chair.GetComponent<ChairController>().IsBusy())
            interactionInfo.text = "Chair is occupied";

        else if(whiteBoard != null && !whiteBoard.isBeingEdited)
            interactionInfo.text = "Press SPACE to start writing on the whiteboard";

        else if (whiteBoard != null && whiteBoard.isBeingEdited && Presenter.Instance.writerID == PhotonNetwork.LocalPlayer.UserId)
            interactionInfo.text = "Press ESC to stop writing";

        else if (whiteBoard != null && whiteBoard.isBeingEdited && Presenter.Instance.writerID != PhotonNetwork.LocalPlayer.UserId)
            interactionInfo.text = "Whiteboard is busy";

        else 
            interactionInfo.text = "";
    }

    [PunRPC]
    public void NotifySitting(bool value, string playerName)
    {
        if (chair != null)
        {
            chair.GetComponent<ChairController>().SetBusy(value);
            chair.GetComponent<ChairController>().playerName = playerName;
        }            
    }

    [PunRPC]
    public void NotifyHandRaisedRPC()
    {
        string msg = GetComponent<PhotonView>().Controller.NickName + " raised a hand!";
        Logger.Instance.LogInfo(msg);
        LogManager.Instance.LogInfo(msg);
    }

    [PunRPC]
    public void NotifySpawnRPC()
    {
        Logger.Instance.LogInfo($"<color=yellow>{GetComponent<PhotonView>().Controller.NickName}</color> just joined the class!");
        LogManager.Instance.LogInfo($"{GetComponent<PhotonView>().Controller.NickName} joined the room");
        GameObject.Find("SpawnAudioSource").GetComponent<AudioSource>().Play();
    }

    [PunRPC]
    public void NotifyTalkRPC(string msg)
    {
        volumeIcon.text = msg;
    }

    [PunRPC]
    public void ClapRPC(bool value)
    {
        clapSound.enabled = value;
    }

    [PunRPC]
    public void LockBoard(bool value, string id, string text)
    {
        if (whiteBoard == null) return;
        whiteBoard.isBeingEdited = value;
        Presenter.Instance.writerID = id;

        if (!value)
        {
            whiteBoard.boardText.text = text;
            LogManager.Instance.LogWhiteboard(text);
        }
    }

}
