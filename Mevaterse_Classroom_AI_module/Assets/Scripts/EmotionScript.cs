using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using TMPro;
using UnityEngine.UI;
using UnityEngine.SceneManagement;


public class EmotionScript : MonoBehaviour
{
    public TextMeshProUGUI happy;
    public TextMeshProUGUI sad;
    public TextMeshProUGUI angry;
    public TextMeshProUGUI surprise;
    public TextMeshProUGUI fear;
    public TextMeshProUGUI neutral;
    public TextMeshProUGUI disgust; 
    public int happyCount=0, sadCount=0, angryCount=0, surpriseCount=0, fearCount=0, neutralCount=0, disgustCount=0;

    private string[] emotions = { "Angry", "Disgust", "Fear", "Happy", "Neutral", "Sad", "Surprise" };

    void Start()
    {
        happy =  GameObject.Find("Happy").GetComponent<TextMeshProUGUI>();
        sad =  GameObject.Find("Sad").GetComponent<TextMeshProUGUI>();
        angry =  GameObject.Find("Angry").GetComponent<TextMeshProUGUI>();
        surprise =  GameObject.Find("Surprise").GetComponent<TextMeshProUGUI>();
        fear =  GameObject.Find("Fear").GetComponent<TextMeshProUGUI>();
        neutral =  GameObject.Find("Neutral").GetComponent<TextMeshProUGUI>();
        disgust =  GameObject.Find("Disgust").GetComponent<TextMeshProUGUI>();
    }

    // Update is called once per frame
    void Update()
    {
        
    }
}
