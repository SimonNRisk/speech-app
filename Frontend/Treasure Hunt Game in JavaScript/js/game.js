function initialise(){
		var obj = new Game();
	}
	

	function Game(){
		//generate map
		//assign first treasure to map
		//score reset to 0
		const digPopup = document.getElementById("dig-popup");
    	digPopup.classList.add("hidden");

		var score = 2000;

		var startBtn = document.getElementById("start-button");
		startBtn.style.display = "none";

		document.getElementById("demo3").innerHTML = score;
		document.getElementById("demo4").innerHTML = "";
		document.getElementById("rules").innerHTML = "";

		var previousgameover = document.getElementById("gameover");

			if(previousgameover){
				previousgameover.remove();
			}

		var gameover = document.createElement("p");
		gameover.id = "gameover";
		gameover.className = "col-sm-12 col-md-12 col-lg-12";
		document.getElementById("board").appendChild(gameover);

		document.getElementById("gameover").innerHTML = "";

		//initialise treasure coordinates
		var x;
		var y;

		var map = [
		    ['A1', 'A2', 'A3', 'A4', 'A5'],
		    ['B1', 'B2', 'B3', 'B4', 'B5'],
		    ['C1', 'C2', 'C3', 'C4', 'C5'],
		    ['D1', 'D2', 'D3', 'D4', 'D5'],
		    ['E1', 'E2', 'E3', 'E4', 'E5']
			];

		function renderMap(){

			var previousgame = document.getElementById("game");

			if(previousgame){
				previousgame.remove();
			}

			var game = document.createElement("div");
			game.id = "game";
			game.className = "col-sm-12 col-md-12 col-lg-12";
			document.getElementById("board").appendChild(game);

			// loop the outer array
			
			for (var i = 0; i < map.length; i++) {
			    // get the size of the inner array
			    var innerArrayLength = map[i].length;
			    // loop the inner array
			    for (var j = 0; j < innerArrayLength; j++) {
			        console.log('[' + i + ',' + j + '] = ' + map[i][j]);
			        var btn = document.createElement("BUTTON");        // Create a <button> element
					btn.className = "game-button";
					btn.id = map[i][j];
					btn.onclick = checkTreasure;
					//var t = document.createTextNode(map[i][j]);       // Create a text node
					//btn.appendChild(t);                                // Append the text to <button>
					document.getElementById("game").appendChild(btn);

					var isMultipleof4 = function (n) 
					{ 
					    if ( n == 0 ){ 
					    	return false; 
						}
						else{
					    	while ( n > 0 )
					    	{ 
					        	n = n - 4; 
						  	}

						    if ( n == 0 ){ 
						        return true; 
						  	}
						  	else{
						    	return false;
						    }
						} 
					}
					
					if ( isMultipleof4(j) == true ){
						var br = document.createElement("div");
						br.className = "clear";
						document.getElementById("game").appendChild(br);
					}
			    }
			}
		}

		function assignTreasure(){
			var m = new renderMap();
			x = Math.floor(Math.random()*map.length);
			y = Math.floor(Math.random()*map.length);

			//document.getElementById("demo").innerHTML = "Current Treasure: " + map[x][y];
			//document.getElementById("demo5").innerHTML = "X: " + x + " Y: " + y;
			
		}
			
		assignTreasure();

		function checkLives(){
			score = score - 250;

			if(score <= 0){

					document.getElementById("gameover").innerHTML = "GAME OVER";
					document.getElementById("gameover").style.color = "red";
					document.getElementById("demo3").innerHTML = score;
					//health.value -= 1;
					var elems = document.getElementsByClassName("game-button");
					for(var i = 0; i < elems.length; i++) {
					    elems[i].disabled = true;
					}
					startBtn.style.display = "block";
				}
				else{
					//health.value -= 1;
					document.getElementById("demo3").innerHTML = score;
				} 
		}

		function checkTreasure(){
			//if coordinates of this button equals cocordinates of current treasure
			//then increase player's score
			//update score and or how close to treasure

			//if dug, allow no other interaction
			if(this.classList.contains("dug")){
				return;
			}

			const digPopup = document.getElementById("dig-popup");
    		const micButton = document.getElementById("mic-button");
			const continueButton = document.getElementById("continue-button");
			const popupText = document.getElementById("popup-text");
    		const targetSquare = this; // Save the clicked square
			
			popupText.textContent = "Pronounce the following word to continue your quest for gold!";
			micButton.classList.remove("hidden");
   		 	continueButton.classList.add("hidden");
			digPopup.classList.remove("hidden");

			micButton.onclick = function () {
				const randomNumber = Math.floor(Math.random() * 2) + 1; // Generate random number 1 or 2

				if (randomNumber === 1) {
					// Correct case
					popupText.textContent = "Correct!";
					micButton.classList.add("hidden"); // Hide the mic button
					continueButton.textContent = "Dig!";
					continueButton.classList.remove("hidden"); // Show the continue button
				} else {
					// Try Again case
					popupText.textContent = "Not quite! Try again...";
					micButton.classList.add("hidden"); // Hide the mic button
					continueButton.textContent = "Try Again"; // Change button text
					continueButton.classList.remove("hidden"); // Show the continue button
				}

				// Continue button functionality based on the result
				continueButton.onclick = function () {
					if (randomNumber === 1) {
						digPopup.classList.add("hidden"); // Hide popup for "Correct"
					} else {
						// Reset to initial state for "Try Again"
						popupText.textContent = "Pronounce the following word to continue your quest for gold!";
						micButton.classList.remove("hidden"); // Show mic button
						continueButton.classList.add("hidden"); // Hide continue button
					}
				};

				//if clicked button is hiding the treasure 
				if(targetSquare.id == map[x][y] && randomNumber == 1){
					score = score + 1000;
					document.getElementById("demo3").innerHTML = score;
					document.getElementById("demo4").innerHTML = "Success! Found some gold!";
					document.getElementById("demo4").style.color = "green"
					//assign another treasure to different location
					document.getElementById("game").remove();
					assignTreasure();
					checkLives();
				}
				//else if clicked button is close to treasure
				else if(randomNumber == 1)
				{ 
						var a = map.length;
						//var b = map[a].length;

						//control variables incase they go out of range of array
						var xplus = x + 1; if(xplus == a){ xplus = a - 1;}
						var xminus = x - 1; if(xminus == -1){ xminus = 0;}
						var yplus = y + 1; if(yplus == a){ yplus = a - 1;}
						var yminus = y - 1; if(yminus == -1){ yminus = 0;}
						
						//specify how close the clicked button is to the treasure  
						var topleft = map[xminus][yminus];
						var topmid = map[xminus][y];
						var topright = map[xminus][yplus];
						var midleft = map[x][yminus];
						var midright = map[x][yplus];
						var bottomleft = map[xplus][yminus];
						var bottommid = map[xplus][y];
						var bottomright = map[xplus][yplus];

						if(targetSquare.id == topleft || 
							targetSquare.id == topmid ||
							targetSquare.id == topright ||
							targetSquare.id == midleft ||
							targetSquare.id == midright ||
							targetSquare.id == bottomleft ||
							targetSquare.id == bottommid ||
							targetSquare.id == bottomright )
						{
							document.getElementById("demo4").innerHTML = "Getting warm...";
							document.getElementById("demo4").style.color = "orange"
							targetSquare.setAttribute("data-proximity", 'warm');
							targetSquare.classList.add("dug");
							checkLives();
						}
						//otherwise tell player they are far away
						else{
							document.getElementById("demo4").innerHTML = "Cold... Keep trying.";
							document.getElementById("demo4").style.color = "red";
							targetSquare.setAttribute("data-proximity", 'cold');
							targetSquare.classList.add("dug");						
							checkLives();
						}
					}
  			}; 

  				
		}
		
		//see hints for dug squares when mouse over
		document.addEventListener("mouseover", function (e) {
			if (e.target.classList.contains("dug")&& score >0) {
				const proximity = e.target.getAttribute("data-proximity");
				document.getElementById("demo4").innerHTML = proximity === "warm" ? "Getting warm..." : "Cold... Keep trying.";
				document.getElementById("demo4").style.color = proximity === "warm" ? "orange" : "red";
			}
		});
		
		document.addEventListener("mouseout", function (e) {
			if (e.target.classList.contains("dug") && score >0) {
				document.getElementById("demo4").innerHTML = ""; 
			}
		});
				
				
				
			
			 
		
	
	}