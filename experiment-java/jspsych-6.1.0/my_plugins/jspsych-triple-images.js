/**
 * jspsych-triple-images
 * Josh de Leeuw
 *
 * plugin for displaying a stimulus and getting a keyboard response
 *
 * documentation: docs.jspsych.org
 *
 **/


jsPsych.plugins["triple-images"] = (function() {

    var plugin = {};

    jsPsych.pluginAPI.registerPreload('triple-images', 'stimulus', 'image');

    plugin.info = {
        name: 'triple-images',
        description: '',
        parameters: {
            disp_func: {
                type: jsPsych.plugins.parameterType.FUNCTION,
                pretty_name: 'Display function',
                default: undefined,
                description: 'The function to be called with the canvas ID. '+
                  'This should handle drawing operations. The return value of this '+
                  'function is stored in trial.stimulus_properties, which is useful '+
                  'for recording particular properties of the stimulus which are '+
                  'only calculated at runtime.'                
            },
            stimulus: {
                type: jsPsych.plugins.parameterType.IMAGE,
                pretty_name: 'Stimulus',
                default: undefined,
                description: 'The images to be displayed'
            },
            radius: {
                type: jsPsych.plugins.parameterType.INT,
                pretty_name: 'Stimulus radius',
                default: 80,
                description: 'Radius of the stimulus.'
            },
            triplet: {
                type: jsPsych.plugins.parameterType.INT, 
                pretty_name: 'Tested triplet',
                default: [0,1,2],
                description: 'The tested triplet.'
            },
            trialId: {
                type: jsPsych.plugins.parameterType.INT,
                pretty_name: 'Id of the tested triplet',
                default: 0,
                description: 'Id of the tested triplet.'+
                'This is required for responses gathering.'
            },
            randomize: {
                type: jsPsych.plugins.parameterType.BOOL,
                pretty_name: 'Left-right randomization',
                default: true,
                description: 'If true, left-right stimuli position '+
                	'are randomized.'
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
                default: 400,
                description: 'Sets the width of the canvas.'
            },
            canvasHeight: {
                type: jsPsych.plugins.parameterType.INT,
                pretty_name: 'Canvas height',
                default: 400,
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
                description: 'Any content here will be displayed below the stimulus.'
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
                description: 'How long to show trial before it ends.'
            },
            response_ends_trial: {
                type: jsPsych.plugins.parameterType.BOOL,
                pretty_name: 'Response ends trial',
                default: true,
                description: 'If true, trial will end when subject makes a response.'
            },
        }
    }

    plugin.trial = function(display_element, trial) {

        var idx = [0,2];
        var seed = [0,1,2,3];
        if(trial.randomize==true){
        	var rdm_idx = jsPsych.randomization.shuffle(idx);
        } else{
        	var rdm_idx = idx;
        } 
        var rdm_seed = jsPsych.randomization.shuffle(seed).slice(0,3);
        
        // display stimulus        
        let canvas = '';
        // Use the supplied HTML for constructing the canvas, if supplied
        if(trial.canvasId !== false) {
            canvas = trial.canvasHTML;
        } else {
            // Otherwise create a new default canvas
            trial.canvasId = '#jspsych-triple-images';
            canvas = '<canvas id="'+trial.canvasId+'" height="'+trial.canvasHeight+
            '" width="'+trial.canvasWidth+'"></canvas>';
        }
        var html = '<img id="im_center" width="512" height="512" src="'
                    +trial.stimulus[trial.triplet[1]]+'"style="display:none">';
        html += '<img id="im_left" width="512" height="512" src="'
                    +trial.stimulus[trial.triplet[rdm_idx[0]]]+'"style="display:none">';
        html += '<img id="im_right" width="512" height="512" src="'
                    +trial.stimulus[trial.triplet[rdm_idx[1]]]+'"style="display:none">';
        html += '<div id="jspsych-triple-images">'+canvas+'</div>';

        if (trial.prompt !== null){
            html += trial.prompt;
        }

        display_element.innerHTML = html;

        // store response
        var response = {
            rt: null,
            key: null,
            stimulus: null
        };
        /*
        console.log(trial.stimulus)
        console.log(trial.triplet)
        console.log(trial.trialId)
        */

        // Execute the supplied drawing function
        response.stimulus = trial.disp_func(trial.canvasId,
                                            "im_center",
                                            "im_left",
                                            "im_right",
                                            trial.radius,
                                            rdm_seed);
        
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
                "stimulus": trial.stimulus,
                "key_press": response.key,
                "idx": rdm_idx
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
            display_element.querySelector('#jspsych-triple-images').className += ' responded';

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
            jsPsych.pluginAPI.setTimeout(
                function() {
                    display_element.querySelector('#jspsych-triple-images').style.visibility = 'hidden';
                },
                trial.stimulus_duration
            );
        }

        // end trial if trial_duration is set
        if (trial.trial_duration !== null) {
            jsPsych.pluginAPI.setTimeout(
                function() {
                end_trial();
                },
                trial.trial_duration
            );
        }

    };

    return plugin;
})();
