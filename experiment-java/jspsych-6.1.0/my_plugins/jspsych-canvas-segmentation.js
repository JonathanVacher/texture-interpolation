/**
 * jspsych-canvas-segmentation
 * a jspsych plugin for free response to questions presented using canvas
 * drawing tools
 *
 * the drawing is done by a function which is supplied as the stimulus.
 * this function is passed the id of the canvas on which it will draw.
 *
 * the canvas can either be supplied as customised HTML, or a default one
 * can be used. If a customised on is supplied, its ID must be specified
 * in a separate variable.
 *
 * Jonathan Vacher - https://github.com/JonathanVacher/ - March 2020
 *
 * documentation: docs.jspsych.org
 *
 */


jsPsych.plugins['canvas-segmentation'] = (function() {

  var plugin = {};

  plugin.info = {
    name: 'canvas-segmentation',
    description: '',
    parameters: {
      stimulus: {
        type: jsPsych.plugins.parameterType.FUNCTION,
        pretty_name: 'Stimulus',
        default: undefined,
        description: 'The function to be called with the canvas ID. '+
          'This should handle drawing operations. The return value of this '+
          'function is stored in trial.stimulus_properties, which is useful '+
          'for recording particular properties of the stimulus which are '+
          'only calculated at runtime.'
      },
      image: {
        type: jsPsych.plugins.parameterType.IMAGE,
        pretty_name: 'Image src',
        default: undefined,
        description: 'The image to be displayed'
      },
      rot: {
        type: jsPsych.plugins.parameterType.FLOAT,
        pretty_name: 'Rotation angle',
        default: 0.0,
        description: 'Rotate the canvas by the given angle.'
      },
      trialId: {
        type: jsPsych.plugins.parameterType.INT,
        pretty_name: 'Id of the tested pair',
        default: 0,
        description: 'Id of the tested pair.'+
            'This is required for reconstruction.'
      },
      xy1: {
        type: jsPsych.plugins.parameterType.INT,
        pretty_name: 'Coordinates of dot 1',
        default: [50,10],
        description: 'First cued location.'
      },
      xy2: {
        type: jsPsych.plugins.parameterType.INT,
        pretty_name: 'Coordinates of dot 2',
        default: [100,10],
        description: 'Second cued location.'
      },
      canvasHTML: {
        type: jsPsych.plugins.parameterType.HTML_STRING,
        pretty_name: 'Canvas HTML',
        default: null,
        description: 'HTML for drawing the canvas. '+
          'Overrides canvas width and height settings.'
      },
      canvasId: {
        type: jsPsych.plugins.parameterType.STRING,
        pretty_name: 'Canvas ID',
        default: false,
        description: 'ID for the canvas. Only necessary when '+
          'supplying canvasHTML. This is required so that the ID '+
          'can be passed to the stimulus function.'
      },
      canvasWidth: {
        type: jsPsych.plugins.parameterType.INT,
        pretty_name: 'Canvas width',
        default: 300,
        description: 'Sets the width of the canvas.'
      },
      canvasHeight: {
        type: jsPsych.plugins.parameterType.INT,
        pretty_name: 'Canvas height',
        default: 150,
        description: 'Sets the height of the canvas.'
      },
      choices: {
        type: jsPsych.plugins.parameterType.KEYCODE,
        array: true,
        pretty_name: 'Choices',
        default: jsPsych.ALL_KEYS,
        description: 'The keys the subject is allowed to press to respond to the stimulus.'
      },
      prompt: {
        type: jsPsych.plugins.parameterType.STRING,
        pretty_name: 'Prompt',
        default: null,
        description: 'Any content here will be displayed below the slider.'
      },
      stimulus_duration: {
        type: jsPsych.plugins.parameterType.INT,
        pretty_name: 'Stimulus duration',
        default: null,
        description: 'How long to hide the stimulus.'
      },
      trial_duration: {
        type: jsPsych.plugins.parameterType.INT,
        pretty_name: 'Trial duration',
        default: null,
        description: 'How long to show the trial.'
      },
      response_ends_trial: {
        type: jsPsych.plugins.parameterType.BOOL,
        pretty_name: 'Response ends trial',
        default: true,
        description: 'If true, trial will end when user makes a response.'
      },
    }
  }

  plugin.trial = function(display_element, trial) {
    let canvas = '';
    // Use the supplied HTML for constructing the canvas, if supplied
    if(trial.canvasId !== false) {
      canvas = trial.canvasHTML;
    } else {
      // Otherwise create a new default canvas
      trial.canvasId = '#jspsych-canvas-keyboard-response-canvas';
      canvas = '<canvas id="'+trial.canvasId+'" height="'+trial.canvasHeight+
        '" width="'+trial.canvasWidth+'"></canvas>';
    }
    var html = '<img id="im_stim" width="256" height="256" src="'+trial.image+'"style="display:none">';

    
    //html += '<div id="jspsych-canvas-keyboard-response-wrapper" style="margin: 100px 0px;">';
    html += '<div id="jspsych-canvas-keyboard-response-stimulus">'+canvas+'</div>';
    
    if (trial.prompt !== null){
      html += trial.prompt;
    }

    display_element.innerHTML = html;

    var response = {
      rt: null,
      key: null,
      stimulus: null,
    };
    
    //console.log(trial.gamma)
    // Execute the supplied drawing function
    response.stimulus = trial.stimulus(trial.canvasId,
                                       trial.xy1, trial.xy2,
                                       "im_stim");//, trial.gamma);
    
    // function to end trial when it is time
    var end_trial = function() {

      // kill any remaining setTimeout handlers
      jsPsych.pluginAPI.clearAllTimeouts();

      // kill keyboard listeners
      if (typeof keyboardListener !== 'undefined') {
        jsPsych.pluginAPI.cancelKeyboardResponse(keyboardListener);
      }

      // gather the data to store for the trial
      var trial_data = {
        "rt": response.rt,
        "pair_1": trial.xy1,
        "pair_2": trial.xy2,
        "pair_id": trial.trialId,
        "key_press": response.key
      };

      // clear the display
      display_element.innerHTML = '';

      // move on to the next trial
      jsPsych.finishTrial(trial_data);
    };

    // function to handle responses by the subject
    var after_response = function(info) {

      // after a valid response, the stimulus will have the CSS class 'responded'
      // which can be used to provide visual feedback that a response was recorded
      display_element.querySelector('#jspsych-canvas-keyboard-response-stimulus').className += ' responded';

      // only record the first response
      if (response.key == null) {
        response = info;
      }

      if (trial.response_ends_trial) {
        end_trial();
      }
    };

    // start the response listener
    if (trial.choices != jsPsych.NO_KEYS) {
      var keyboardListener = jsPsych.pluginAPI.getKeyboardResponse({
        callback_function: after_response,
        valid_responses: trial.choices,
        rt_method: 'performance',
        persist: false,
        allow_held_key: false
      });
    }

    // hide stimulus if stimulus_duration is set
    if (trial.stimulus_duration !== null) {
      jsPsych.pluginAPI.setTimeout(function() {
        display_element.querySelector('#jspsych-canvas-keyboard-response-stimulus').style.visibility = 'hidden';
      }, trial.stimulus_duration);
    }

    // end trial if trial_duration is set
    if (trial.trial_duration !== null) {
      jsPsych.pluginAPI.setTimeout(function() {
        end_trial();
      }, trial.trial_duration);
    }


    var startTime = (new Date()).getTime();
  };

  return plugin;
})();
