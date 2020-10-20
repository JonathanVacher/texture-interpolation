var images_gauss = [[],[],[]];
var im_num = ['15','108','132'];
var n = 11; 
for (var i=0; i<3; i++){
    for (var j=0; j<n; j++){
        if (j<9){
            images_gauss[i].push('img/Im_'+im_num[i]+'/Im_'+im_num[i]+'_interp_0'+(j+1).toString()+'.png');
        } 
        else{
            images_gauss[i].push('img/Im_'+im_num[i]+'/Im_'+im_num[i]+'_interp_'+(j+1).toString()+'.png');
        }   
    }
}

var images_arbitrary = [[],[],[],[],[],[]];
var pair_num = ['03','06','13'];
var pair_num_save = ['03_w','06_w','13_w'];

for (var i=0; i<3; i++){
    for (var j=0; j<n; j++){
        if (j<9){
            images_arbitrary[i].push('img/pair'+pair_num[i]+'_wasser'
                    +'/pair'+pair_num[i]+'_interp_0'+(j+1).toString()+'.png');
        } 
        else{
            images_arbitrary[i].push('img/pair'+pair_num[i]+'_wasser'
                    +'/pair'+pair_num[i]+'_interp_'+(j+1).toString()+'.png');
        }   
    }
}

// test
//var comb = [[0,2,4],[0,8,10]];
// full
var comb = [[0,2,4],[0,2,5],[0,2,6],[0,2,7],[0,2,8],[0,2,9],[0,2,10],[0,3,5],[0,3,6],[0,3,7],[0,3,8],[0,3,9],[0,3,10],[0,4,6],[0,4,7],[0,4,8],[0,4,9],[0,4,10],[0,5,7],[0,5,8],[0,5,9],[0,5,10],[0,6,8],[0,6,9],[0,6,10],[0,7,9],[0,7,10],[0,8,10],[1,3,5],[1,3,6],[1,3,7],[1,3,8],[1,3,9],[1,3,10],[1,4,6],[1,4,7],[1,4,8],[1,4,9],[1,4,10],[1,5,7],[1,5,8],[1,5,9],[1,5,10],[1,6,8],[1,6,9],[1,6,10],[1,7,9],[1,7,10],[1,8,10],[2,4,6],[2,4,7],[2,4,8],[2,4,9],[2,4,10],[2,5,7],[2,5,8],[2,5,9],[2,5,10],[2,6,8],[2,6,9],[2,6,10],[2,7,9],[2,7,10],[2,8,10],[3,5,7],[3,5,8],[3,5,9],[3,5,10],[3,6,8],[3,6,9],[3,6,10],[3,7,9],[3,7,10],[3,8,10],[4,6,8],[4,6,9],[4,6,10],[4,7,9],[4,7,10],[4,8,10],[5,7,9],[5,7,10],[5,8,10],[6,8,10]];
var comb_train = [[0,2,10],[0,8,10],[1,3,10],[0,7,9],[0,6,8],[2,4,10],[3,5,10],[0,5,7],[0,4,6],[4,6,10]];

var n_comb = comb.length; 
var triplets = [];
for (var i=0; i<n_comb; i++){
    triplets.push({id_and_triplet: {trialId: i, triplet: comb[i]}});
}

var n_comb_train = comb_train.length; 
var triplets_train = [];
for (var i=0; i<n_comb_train; i++){
    triplets_train.push({id_and_triplet: {trialId: i, triplet: comb_train[i]}});
}



/* Functions */

var rdm_block_idx = jsPsych.randomization.shuffle([0,1,2]);
var im_idx = rdm_block_idx[0];
var id = 0;
var comments = 0;
var n_pause = 42; // 42 in true experiment
var feedback_count = 0;
var show_intro_practice = 1;
var gamma_val = 2.2; 
var gamma_val1 = 1.0;
var gamma_val2 = 1.0;
var gamma_val3 = 1.0;
var grey = parseInt(127);
var screen_type_options = ["The screen attached to the laptop I'm using",
			    "An external screen plugged to my laptop/desktop"];


function measureGamma1(canvasId, gamma){
	let c = document.getElementById(canvasId);
	let ctx = c.getContext("2d");
	let img1 = new Image;
	img1.src = 'img/stripes-patches-1.png';

	let g1 = parseFloat(255*Math.pow((Math.pow(0.196, gamma)+
		                      Math.pow(0.392, gamma)+
		                      Math.pow(0.588, gamma))/4,
		                      1/gamma));    

	ctx.drawImage(img1, 384, 0);    

	ctx.beginPath();
	ctx.arc(384+128,128, 70, 0, 2*Math.PI, true);
	ctx.fillStyle = "rgb("+g1+","+g1+","+g1+")";
	ctx.fill();
}

function measureGamma2(canvasId, gamma){
	let c = document.getElementById(canvasId);
	let ctx = c.getContext("2d");
	let img1 = new Image;
	img1.src = 'img/stripes-patches-2.png';

	let g1 = parseFloat(255*Math.pow((Math.pow(0.251, gamma)+
		                      Math.pow(0.749, gamma)+
		                      1.0)/4,
		                      1/gamma));
		                      
	ctx.drawImage(img1, 384, 0);    

	ctx.beginPath();
	ctx.arc(384+128,128, 70, 0, 2*Math.PI, true);
	ctx.fillStyle = "rgb("+g1+","+g1+","+g1+")";
	ctx.fill();
}

function measureGamma3(canvasId, gamma){
	let c = document.getElementById(canvasId);
	let ctx = c.getContext("2d");
	let img1 = new Image;
	img1.src = 'img/stripes-patches-3.png';

	let g1 = parseFloat(255*Math.pow((Math.pow(0.216, gamma)+
		                      Math.pow(0.412, gamma)+
		                      Math.pow(0.608, gamma)+
		                      Math.pow(0.804, gamma))/4,
		                      1/gamma));    

	ctx.drawImage(img1, 384, 0);    

	ctx.beginPath();
	ctx.arc(384+128,128, 70, 0, 2*Math.PI, true);
	ctx.fillStyle = "rgb("+g1+","+g1+","+g1+")";
	ctx.fill();
}


function measureGamma(canvasId, gamma){
	let c = document.getElementById(canvasId);
	let ctx = c.getContext("2d");
	let img1 = new Image;
	let img2 = new Image;
	let img3 = new Image;
	img1.src = 'img/stripes-patches-1.png';
	img2.src = 'img/stripes-patches-2.png';
	img3.src = 'img/stripes-patches-3.png';

	let g1 = parseFloat(255*Math.pow((Math.pow(0.196, gamma)+
		                      Math.pow(0.392, gamma)+
		                      Math.pow(0.588, gamma))/4,
		                      1/gamma));    
	let g2 = parseFloat(255*Math.pow((Math.pow(0.251, gamma)+
		                      Math.pow(0.749, gamma)+
		                      1.0)/4,
		                      1/gamma));    
	let g3 = parseFloat(255*Math.pow((Math.pow(0.216, gamma)+
		                      Math.pow(0.412, gamma)+
		                      Math.pow(0.608, gamma)+
		                      Math.pow(0.804, gamma))/4,
		                      1/gamma));    

	ctx.drawImage(img1, 0, 0);    
	ctx.drawImage(img2, 384, 0);    
	ctx.drawImage(img3, 768, 0);    

	ctx.beginPath();
	ctx.arc(128,128, 70, 0, 2*Math.PI, true);
	ctx.fillStyle = "rgb("+g1+","+g1+","+g1+")";
	ctx.fill();

	ctx.beginPath();
	ctx.arc(384+128,128, 70, 0, 2*Math.PI, true);
	ctx.fillStyle = "rgb("+g2+","+g2+","+g2+")";
	ctx.fill();

	ctx.beginPath();
	ctx.arc(768+128,128, 70, 0, 2*Math.PI, true);         
	ctx.fillStyle = "rgb("+g3+","+g3+","+g3+")";
	ctx.fill();
}

function adjustGamma(data) {
	const gammaCorrection = 1 / gamma_val;                    
	for (var i = 0; i < data.length; i += 4) {
		data[i] = 255 * Math.pow((data[i] / 255), gammaCorrection);
		data[i+1] = 255 * Math.pow((data[i+1] / 255), gammaCorrection);
		data[i+2] = 255 * Math.pow((data[i+2] / 255), gammaCorrection);
	}
}

function intro_practice_cond(){
	if(show_intro_practice==1){
		show_intro_practice = 0;
		return true;
	} else {
		return false;
	}
}



function feedback_stim(){
	let resp = jsPsych.data.get()
			.last(1).select('key_press');
	if (feedback_count==0 || feedback_count==2){
		if (resp.values[0]==39){
			return "Wrong."
		} else 
		{
			return "Correct!"
		}
	} else
	{
		if (resp.values[0]==37){
			return "Wrong."
		} else 
		{
			return "Correct!"
		}
	}
}

function feedback_cond(){
	if(feedback_count<3){
	    return true;
	} else {
	    return false;
	}
}

function practice_loop(data){
	let last_10_trials = jsPsych.data.get()
			.filter({save_type: 'response_train'})
			.last(10)
			.select('key_press');
	let q1 = last_10_trials.values[0];
	let q2 = last_10_trials.values[1];
	let q3 = last_10_trials.values[2];
	let q = (q1==37)*(q2==39)*(q3==37);
	if (q==true){
		return false;
	}
	else{
		show_intro_practice = 1;
		feedback_count = 0;
		return true;
	}
}

function pause_stim(){
	let count = jsPsych.data.get()
			.filter({save_type: 'response',
				 texture_id: im_num[im_idx]}).count();
	let count_run = Math.floor(count/n_pause);
	return "Run "+count_run.toString()+"/8. Take a short break (<1 min) if needed"
	     + " and press Enter to continue!"
}

function pause_cond(){
	let count = jsPsych.data.get()
	       .filter({save_type: 'response'}).count();
	if(count%n_pause == 0){
	    return true;
	} else {
	    return false;
	}
}
                            
function save_data(data){
	delete data.stimulus
	delete data.trial_type
	delete data.trial_index
	var idx_ = jsPsych.data.get().last(2).select('idx');
	data.triplet_idx =
		jsPsych.timelineVariable('id_and_triplet',
			                 true).trialId  
	data.lr_idx = idx_.values[0];
	data.participant_id = id;
	data.texture_id = im_num[im_idx];
	if(data.key_press == 37){
	    if(data.lr_idx[0] == 0){
		data.mlds_resp = false;
	    } 
	    else {
		data.mlds_resp = true;
	    }
	}
	else if(data.key_press == 39) {
	    if(data.lr_idx[0] == 0){
		data.mlds_resp = true;
	    } 
	    else {
		data.mlds_resp = false;
	    }
	}
	var count = jsPsych.data.get()
	       .filter({save_type: 'response'}).count();
	if(count%n_pause == 0){
	    data.pause = true;
	} else {
	    data.pause = false;
	}                      
}

function disp_train(canvas, im_center, im_left, im_right, radius, rdm_seed){
	var c = document.getElementById(canvas);
	var ctx = c.getContext("2d");
	var img_c = document.getElementById(im_center);
	var img_l = document.getElementById(im_left);
	var img_r = document.getElementById(im_right);
	var c_w = c.width;
	var c_h = c.height;
	var half_w = img_c.width/2;
	var half_h = img_c.height/2;
	var pos = [[0.5,0.5],[0.5,0.5],[0.5,0.5],[0.5,0.5]];

	var posx_c = 0.5*c_w;
	var posy_c = (0.5-0.3)*c_h;

	var posx_l = (0.5-0.3*Math.sqrt(3)/2)*c_w;
	var posy_l = (0.5+0.3*0.5)*c_h; 

	var posx_r = (0.5+0.3*Math.sqrt(3)/2)*c_w;
	var posy_r = (0.5+0.3*0.5)*c_h;

	var grad_c = ctx.createRadialGradient(posx_c,posy_c,radius-10,
		                          posx_c,posy_c,radius-0);
	grad_c.addColorStop(0, 'rgba(127,127,127,0.0)');
	grad_c.addColorStop(1, 'rgba(127,127,127,1.0)');

	var grad_l = ctx.createRadialGradient(posx_l,posy_l,radius-10,
		                          posx_l,posy_l,radius-0);
	grad_l.addColorStop(0, 'rgba(127,127,127,0.0)');
	grad_l.addColorStop(1, 'rgba(127,127,127,1.0)');

	var grad_r = ctx.createRadialGradient(posx_r,posy_r,radius-10,
		                          posx_r,posy_r,radius-0);
	grad_r.addColorStop(0, 'rgba(127,127,127,0.0)');
	grad_r.addColorStop(1, 'rgba(127,127,127,1.0)');

	//
	ctx.save(); 
	ctx.beginPath();
	ctx.arc(posx_c, posy_c, 0.99*radius, 0, 6.28, false);            
	ctx.closePath();            
	ctx.clip(); 
	ctx.drawImage(img_c, -128*pos[rdm_seed[0]][0]-half_w+posx_c,
		         -128*pos[rdm_seed[0]][1]-half_h+posy_c);
	ctx.restore();

	ctx.save();
	ctx.beginPath();            
	ctx.arc(posx_c, posy_c, 1*radius, 0, 6.28, false);
	ctx.closePath();
	ctx.clip();
	ctx.fillStyle = grad_c;
	ctx.fillRect(0,0,c_w,c_h);  
	ctx.restore();

	//
	ctx.save();             
	ctx.beginPath();
	ctx.arc(posx_l, posy_l, 0.99*radius, 0, 6.28, false);            
	ctx.closePath();
	ctx.clip(); 
	ctx.drawImage(img_l, -128*pos[rdm_seed[1]][0]-half_w+posx_l,
		         -128*pos[rdm_seed[1]][1]-half_h+posy_l);
	ctx.restore();

	ctx.save();
	ctx.beginPath();            
	ctx.arc(posx_l, posy_l, 1*radius, 0, 6.28, false);
	ctx.closePath();
	ctx.clip();
	ctx.fillStyle = grad_l;
	ctx.fillRect(0,0,c_w,c_h);
	ctx.restore();

	//
	ctx.save(); 
	ctx.beginPath();
	ctx.arc(posx_r, posy_r, 0.99*radius, 0, 6.28, false);
	ctx.closePath();
	ctx.clip(); 
	ctx.drawImage(img_r, -128*pos[rdm_seed[2]][0]-half_w+posx_r,
		         -128*pos[rdm_seed[2]][1]-half_h+posy_r);
	ctx.restore();

	ctx.save();
	ctx.beginPath();            
	ctx.arc(posx_r, posy_r, 1*radius, 0, 6.28, false);
	ctx.closePath();
	ctx.clip();
	ctx.fillStyle = grad_r;
	ctx.fillRect(0,0,c_w,c_h);
	ctx.restore();


	let all_data = ctx.getImageData(0, 0, c_w, c_h);
	adjustGamma(all_data.data);
	ctx.putImageData(all_data, 0, 0);    
}


function disp(canvas, im_center, im_left, im_right, radius, rdm_seed){
	var c = document.getElementById(canvas);
	var ctx = c.getContext("2d");
	var img_c = document.getElementById(im_center);
	var img_l = document.getElementById(im_left);
	var img_r = document.getElementById(im_right);
	var c_w = c.width;
	var c_h = c.height;
	var half_w = img_c.width/2;
	var half_h = img_c.height/2;
	var pos = [[0,0],[0,1],[1,0],[1,1]];

	var posx_c = 0.5*c_w;
	var posy_c = (0.5-0.3)*c_h;

	var posx_l = (0.5-0.3*Math.sqrt(3)/2)*c_w;
	var posy_l = (0.5+0.3*0.5)*c_h; 

	var posx_r = (0.5+0.3*Math.sqrt(3)/2)*c_w;
	var posy_r = (0.5+0.3*0.5)*c_h;

	var grad_c = ctx.createRadialGradient(posx_c,posy_c,radius-10,
		                          posx_c,posy_c,radius-0);
	grad_c.addColorStop(0, 'rgba(127,127,127,0.0)');
	grad_c.addColorStop(1, 'rgba(127,127,127,1.0)');

	var grad_l = ctx.createRadialGradient(posx_l,posy_l,radius-10,
		                          posx_l,posy_l,radius-0);
	grad_l.addColorStop(0, 'rgba(127,127,127,0.0)');
	grad_l.addColorStop(1, 'rgba(127,127,127,1.0)');

	var grad_r = ctx.createRadialGradient(posx_r,posy_r,radius-10,
		                          posx_r,posy_r,radius-0);
	grad_r.addColorStop(0, 'rgba(127,127,127,0.0)');
	grad_r.addColorStop(1, 'rgba(127,127,127,1.0)');

	//
	ctx.save(); 
	ctx.beginPath();
	ctx.arc(posx_c, posy_c, 0.99*radius, 0, 6.28, false);            
	ctx.closePath();            
	ctx.clip(); 
	ctx.drawImage(img_c, -128*pos[rdm_seed[0]][0]-half_w+posx_c,
		         -128*pos[rdm_seed[0]][1]-half_h+posy_c);
	ctx.restore();

	ctx.save();
	ctx.beginPath();            
	ctx.arc(posx_c, posy_c, 1*radius, 0, 6.28, false);
	ctx.closePath();
	ctx.clip();
	ctx.fillStyle = grad_c;
	ctx.fillRect(0,0,c_w,c_h);  
	ctx.restore();

	//
	ctx.save();             
	ctx.beginPath();
	ctx.arc(posx_l, posy_l, 0.99*radius, 0, 6.28, false);            
	ctx.closePath();
	ctx.clip(); 
	ctx.drawImage(img_l, -128*pos[rdm_seed[1]][0]-half_w+posx_l,
		         -128*pos[rdm_seed[1]][1]-half_h+posy_l);
	ctx.restore();

	ctx.save();
	ctx.beginPath();            
	ctx.arc(posx_l, posy_l, 1*radius, 0, 6.28, false);
	ctx.closePath();
	ctx.clip();
	ctx.fillStyle = grad_l;
	ctx.fillRect(0,0,c_w,c_h);
	ctx.restore();

	//
	ctx.save(); 
	ctx.beginPath();
	ctx.arc(posx_r, posy_r, 0.99*radius, 0, 6.28, false);
	ctx.closePath();
	ctx.clip(); 
	ctx.drawImage(img_r, -128*pos[rdm_seed[2]][0]-half_w+posx_r,
		         -128*pos[rdm_seed[2]][1]-half_h+posy_r);
	ctx.restore();

	ctx.save();
	ctx.beginPath();            
	ctx.arc(posx_r, posy_r, 1*radius, 0, 6.28, false);
	ctx.closePath();
	ctx.clip();
	ctx.fillStyle = grad_r;
	ctx.fillRect(0,0,c_w,c_h);
	ctx.restore();


	let all_data = ctx.getImageData(0, 0, c_w, c_h);
	adjustGamma(all_data.data);
	ctx.putImageData(all_data, 0, 0);    
}




