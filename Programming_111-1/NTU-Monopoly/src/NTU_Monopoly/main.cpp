#include <SDL.h>
#include <SDL_image.h>
#include <SDL_ttf.h>
#include <stdio.h>
#include <string>
#include <iostream>
#include <iomanip>
#include <time.h>
#include <stdlib.h>
#include "card.h"
#include "text.h"
using namespace std;
SDL_Rect card_rect;
SDL_Rect p1_rect;
bool printed = false;
bool rolled = false;
bool roll_done = false;
bool turn_complete = false;
bool mortage_done = false;//Sorry, "mortgage" is misspelled in all variable names
bool mortage_print = false;
bool enter = false;
int after_roll_pos;
int dice;
int mortage_index = 0;
bool start = false;


//Starts up SDL and creates window
bool init();

//Loads media
bool loadMedia();

//Frees media and shuts down SDL
void close();

//Loads individual image
SDL_Surface* loadSurface( std::string path );

SDL_Texture* loadTexture( std::string path );

//The window we'll be rendering to
SDL_Window* gWindow = NULL;

//The surface contained by the window
SDL_Surface* gScreenSurface = NULL;


//Current displayed image
SDL_Surface* gCurrentSurface = NULL;

//The window renderer

//Current displayed texture
SDL_Texture* monopoly_board = NULL;


SDL_Color color_test = {0,0xFA,0,0xFA};
bool init()
{
	//Initialization flag
	bool success = true;

	//Initialize SDL
	if( SDL_Init( SDL_INIT_VIDEO ) < 0 )
	{
		printf( "SDL could not initialize! SDL Error: %s\n", SDL_GetError() );
		success = false;
	}
	else
	{
		//Create window
		gWindow = SDL_CreateWindow( "NTU Monopoly", SDL_WINDOWPOS_UNDEFINED, SDL_WINDOWPOS_UNDEFINED, SCREEN_WIDTH, SCREEN_HEIGHT, SDL_WINDOW_SHOWN );
		if( gWindow == NULL )
		{
			printf( "Window could not be created! SDL Error: %s\n", SDL_GetError() );
			success = false;
		}
		else
		{
			//Get window surface
			gScreenSurface = SDL_GetWindowSurface( gWindow );
			gRenderer = SDL_CreateRenderer( gWindow, -1, SDL_RENDERER_ACCELERATED );
            if( gRenderer == NULL )
            {
                printf( "Renderer could not be created! SDL Error: %s\n", SDL_GetError() );
                success = false;
            }
            else
            {
                //Initialize renderer color
                SDL_SetRenderDrawColor( gRenderer, 0xFF, 0xFF, 0xFF, 0xFF );

                //Initialize PNG loading
                int imgFlags = IMG_INIT_PNG;
                if( !( IMG_Init( imgFlags ) & imgFlags ) )
                {
                    printf( "SDL_image could not initialize! SDL_image Error: %s\n", IMG_GetError() );
                    success = false;
                }
                if( TTF_Init() == -1 )
				{
					printf( "SDL_ttf could not initialize! SDL_ttf Error: %s\n", TTF_GetError() );
					success = false;
				}
            }
		}
	}

	return success;
}

bool loadMedia()
{
	//Loading success flag
	bool success = true;
    monopoly_board = loadTexture( "board.png" );
    if( monopoly_board == NULL )
    {
        printf( "Failed to load texture image!\n" );
        success = false;
    }
    gFont = TTF_OpenFont( "OpenSans-Bold.ttf", 32);
	if( gFont == NULL )
	{
		printf( "Failed to load lazy font! SDL_ttf Error: %s\n", TTF_GetError() );
		success = false;
	}
	if( !chance_0.loadFromFile( "chance/chance.png" ) )
	{
		printf( "Failed to load sprite sheet texture!\n" );
		success = false;
	}
	if( !chance_1.loadFromFile( "chance/chance (1).png" ) )
	{
		printf( "Failed to load sprite sheet texture!\n" );
		success = false;
	}
	if( !chance_2.loadFromFile( "chance/chance (2).png" ) )
	{
		printf( "Failed to load sprite sheet texture!\n" );
		success = false;
	}
	if( !chance_3.loadFromFile( "chance/chance (3).png" ) )
	{
		printf( "Failed to load sprite sheet texture!\n" );
		success = false;
	}
	if( !chance_4.loadFromFile( "chance/chance (4).png" ) )
	{
		printf( "Failed to load sprite sheet texture!\n" );
		success = false;
	}
	if( !chance_5.loadFromFile( "chance/chance (5).png" ) )
	{
		printf( "Failed to load sprite sheet texture!\n" );
		success = false;
	}
	if( !chance_6.loadFromFile( "chance/chance (6).png" ) )
	{
		printf( "Failed to load sprite sheet texture!\n" );
		success = false;
	}
	if( !chance_7.loadFromFile( "chance/chance (7).png" ) )
	{
		printf( "Failed to load sprite sheet texture!\n" );
		success = false;
	}
	if( !chance_8.loadFromFile( "chance/chance (8).png" ) )
	{
		printf( "Failed to load sprite sheet texture!\n" );
		success = false;
	}
	if( !chance_9.loadFromFile( "chance/chance (9).png" ) )
	{
		printf( "Failed to load sprite sheet texture!\n" );
		success = false;
	}
	if( !chance_10.loadFromFile( "chance/chance (10).png" ) )
	{
		printf( "Failed to load sprite sheet texture!\n" );
		success = false;
	}
	if( !chance_11.loadFromFile( "chance/chance (11).png" ) )
	{
		printf( "Failed to load sprite sheet texture!\n" );
		success = false;
	}
	if( !chance_12.loadFromFile( "chance/chance (12).png" ) )
	{
		printf( "Failed to load sprite sheet texture!\n" );
		success = false;
	}
	if( !chance_13.loadFromFile( "chance/chance (13).png" ) )
	{
		printf( "Failed to load sprite sheet texture!\n" );
		success = false;
	}
	if( !chance_14.loadFromFile( "chance/chance (14).png" ) )
	{
		printf( "Failed to load sprite sheet texture!\n" );
		success = false;
	}
	if( !chance_15.loadFromFile( "chance/chance (15).png" ) )
	{
		printf( "Failed to load sprite sheet texture!\n" );
		success = false;
	}
	if( !chance_16.loadFromFile( "chance/chance (16).png" ) )
	{
		printf( "Failed to load sprite sheet texture!\n" );
		success = false;
	}
	if (!community_0.loadFromFile( "community/community (1).png"))
    {
		printf( "Failed to load sprite sheet texture!\n" );
		success = false;
	}
	if (!community_1.loadFromFile( "community/community (2).png"))
    {
		printf( "Failed to load sprite sheet texture!\n" );
		success = false;
	}
	if (!community_2.loadFromFile( "community/community (3).png"))
    {
		printf( "Failed to load sprite sheet texture!\n" );
		success = false;
	}
	if (!community_3.loadFromFile( "community/community (4).png"))
    {
		printf( "Failed to load sprite sheet texture!\n" );
		success = false;
	}
	if (!community_4.loadFromFile( "community/community (5).png"))
    {
		printf( "Failed to load sprite sheet texture!\n" );
		success = false;
	}
	if (!community_5.loadFromFile( "community/community (6).png"))
    {
		printf( "Failed to load sprite sheet texture!\n" );
		success = false;
	}
	if (!community_6.loadFromFile( "community/community (7).png"))
    {
		printf( "Failed to load sprite sheet texture!\n" );
		success = false;
	}
	if (!community_7.loadFromFile( "community/community (8).png"))
    {
		printf( "Failed to load sprite sheet texture!\n" );
		success = false;
	}
	if (!community_8.loadFromFile( "community/community (9).png"))
    {
		printf( "Failed to load sprite sheet texture!\n" );
		success = false;
	}
	if (!community_9.loadFromFile( "community/community (10).png"))
    {
		printf( "Failed to load sprite sheet texture!\n" );
		success = false;
	}
	if (!community_10.loadFromFile( "community/community (11).png"))
    {
		printf( "Failed to load sprite sheet texture!\n" );
		success = false;
	}
	if (!community_11.loadFromFile( "community/community (12).png"))
    {
		printf( "Failed to load sprite sheet texture!\n" );
		success = false;
	}
	if (!community_12.loadFromFile( "community/community (13).png"))
    {
		printf( "Failed to load sprite sheet texture!\n" );
		success = false;
	}
	if (!community_13.loadFromFile( "community/community (14).png"))
    {
		printf( "Failed to load sprite sheet texture!\n" );
		success = false;
	}
	if (!community_14.loadFromFile( "community/community (15).png"))
    {
		printf( "Failed to load sprite sheet texture!\n" );
		success = false;
	}
	if (!community_15.loadFromFile( "community/community (16).png"))
    {
		printf( "Failed to load sprite sheet texture!\n" );
		success = false;
	}
	if (!community_16.loadFromFile( "community/community (17).png"))
    {
		printf( "Failed to load sprite sheet texture!\n" );
		success = false;
	}
	else
	{
		card_rect.x = 0;
        card_rect.y = 0;
        card_rect.h = 500;
        card_rect.w = 500;

        p1_rect.x = 0;
        p1_rect.y = 0;
        p1_rect.h = 500;
        p1_rect.w = 500;
	}
	if( !p1.loadFromFile( "characters/character1.png" ) )
	{
		printf( "Failed to load sprite sheet texture!\n" );
		success = false;
	}
	if( !p2.loadFromFile( "characters/character2.png" ) )
	{
		printf( "Failed to load sprite sheet texture!\n" );
		success = false;
	}
	if( !p3.loadFromFile( "characters/character3.png" ) )
	{
		printf( "Failed to load sprite sheet texture!\n" );
		success = false;
	}
	if( !p4.loadFromFile( "characters/character4.png" ) )
	{
		printf( "Failed to load sprite sheet texture!\n" );
		success = false;
	}
	if( !flag.loadFromFile( "flags/flag.png" ) )
	{
		printf( "Failed to load sprite sheet texture!\n" );
		success = false;
	}
	if( !superbig.loadFromFile( "flags/superbig.png" ) )
	{
		printf( "Failed to load sprite sheet texture!\n" );
		success = false;
	}
	return success;
}

void close()
{
	//Deallocate surfaces

	//Destroy window
	SDL_DestroyWindow( gWindow );
	gWindow = NULL;

	//Quit SDL subsystems
	SDL_Quit();
}

SDL_Surface* loadSurface( std::string path )
{
	//Load image at specified path
	SDL_Surface* loadedSurface = SDL_LoadBMP( path.c_str() );
	if( loadedSurface == NULL )
	{
		printf( "Unable to load image %s! SDL Error: %s\n", path.c_str(), SDL_GetError() );
	}

	return loadedSurface;
}

void turn(player&);
void display(player&);
void roll_dice(player&);
int main( int argc, char* args[] )
{
    if( !init() )
	{
		printf( "Failed to initialize!\n" );
	}
	else
	{
		//Load media
		if( !loadMedia() )
		{
			printf( "Failed to load media!\n" );
		}
		else
		{
			//Main loop flag
			bool quit = false;
			//Event handler
			SDL_Event e;
            vector <player*> player_list = {&P1, &P2, &P3, &P4};
            int who = 0;//which player's turn
            SDL_Color textColor = { 0, 0, 0xFF, 0xFF };//blue
			while( !quit)
			{

				//Handle events on queue
				while( SDL_PollEvent( &e ) != 0 )
				{
					//User requests quit
					if( e.type == SDL_QUIT )
					{
						quit = true;
					}
					else if( e.type == SDL_KEYDOWN )
					{
						switch( e.key.keysym.sym )
						{
                            case SDLK_d://roll dice
                                rolled = true;
                                break;

                            case SDLK_y://the 'yes' option when purchasing or mortgaging
                                yes = true;
                                break;

                            case SDLK_n://the 'no' option when purchasing or mortgaging
                                no = true;
                                break;

                            case SDLK_a://press a to switch turns at the end of a player's turn
                                enter = true;
                                break;
						}
					}
				}
                turn_message = "Turn:";
                money_message = "Balance:";
                dice_message = " ";
                game1_message = " ";
                game2_message = " ";
                game3_message = " ";
                mortage1_message = " ";
                mortage2_message = " ";
                over_message = " ";
                card_texture = NULL;
				if (player::bankrupt_count < 3)//if 3 players have gone bankrupt, it means that only one player is remaining, hence game over
                    turn(*player_list[who]);
                else
                {
                    turn_message = "Game Over!";
                    string winner;
                    if (!P1.bankrupt) winner = P1.getname();
                    else if (!P2.bankrupt) winner = P2.getname();
                    else if (!P3.bankrupt) winner = P3.getname();
                    else if (!P4.bankrupt) winner = P4.getname();
                    money_message = winner + " has won!";
                }
				SDL_RenderClear( gRenderer );
				SDL_RenderCopy( gRenderer, monopoly_board, NULL, NULL );

				player_text.loadFromRenderedText( turn_message.c_str(), textColor );
                money_text.loadFromRenderedText( money_message.c_str(), textColor );
                dice_text.loadFromRenderedText( dice_message.c_str(), textColor );
                game1_text.loadFromRenderedText( game1_message.c_str(), textColor );
                game2_text.loadFromRenderedText( game2_message.c_str(), textColor );
                game3_text.loadFromRenderedText( game3_message.c_str(), textColor );
                mortage1_text.loadFromRenderedText( mortage1_message.c_str(), textColor );
                mortage2_text.loadFromRenderedText( mortage2_message.c_str(), textColor );
                over_text.loadFromRenderedText( over_message.c_str(), textColor );

                if (card_texture != NULL)//card_texture points to the texture of the chance or community chest card that has been drawn
                    card_texture->render(150,150, &card_rect);
                if (!P1.bankrupt)
                    p1.render(P1.posx(),P1.posy());
                if (!P2.bankrupt)
                    p2.render(P2.posx(),P2.posy());
                if (!P3.bankrupt)
                    p3.render(P3.posx(),P3.posy());
                if (!P4.bankrupt)
                    p4.render(P4.posx(),P4.posy());

                for (int i = 0; i < 40; i++)
                {
                    property* ptr = dynamic_cast<property*> (board[i]);
                    if (ptr != NULL)//if the block is a property block
                    {
                        ptr->flag_render();
                    }
                }
                //rendering all the blue game information texts shown in the middle of the screen
                player_text.render( 130, 110 );
                money_text.render( 130, 155 );
                dice_text.render( 130, 200);
                game1_text.render(130, 245);
                game2_text.render(130, 290);
                game3_text.render(130, 335);
                mortage1_text.render(130, 380);
                mortage2_text.render(130,425);
                over_text.render(130, 625);

				//Update screen
				SDL_RenderPresent( gRenderer );
				enter = false;
                if (turn_complete)
                {
                    who++;
                    if (who == 4) who = 0;
                    //reset all flags
                    printed = false;
                    rolled = false;
                    roll_done = false;
                    turn_complete = false;
                    mortage_done = false;
                    mortage_print = false;
                    blk_printed = false;
                    blk_run = false;
                    yes = false;
                    no = false;
                    flag1 = false;
                    flag2 = false;
                    mortage_index = 0;
                }
			}
		}
	}
    close();
	return 0;
}
void display(player &P)
{
    if (P.skip != 0)
    {
        game1_message =  P.getname() + " skips this turn.";
        if (!printed) P.skip --;
        printed = true;
        return;
    }
    game1_message =  "Pos:" + board[P.pos]->getname();
    if (!roll_done) dice_message = "Please press 'D' to roll dice......" ;
    else dice_message = "You rolled " + to_string (dice);
    printed = true;
}
void roll_dice(player &P)
{
    if (roll_done) return;
    srand(time(0));
    dice = rand() % 6 + 1;
    for (int i = 0; i < dice; i++)
    {
        P.pos++;
        if (P.pos == 40)//passes MRT Gongguan
        {
            P.pos = 0;
            P.money += 5000;
        }
    }
    after_roll_pos = P.pos;
    roll_done = true;
}
void mortage_check(player &P)
{
    if (P.money >= 0)//no need to mortgage
    {
        mortage_done = true;
        return;
    }
    bool bankrupt = true;
    for (int i = 0; i < 40; i++)
    {
        property* checker = dynamic_cast<property*>(board[i]);
        if (checker != NULL && checker->owner != NULL && *checker->owner == P)//if the player still owns any property
        {
            bankrupt = false;
            break;
        }
    }
    if (bankrupt)
    {
        mortage1_message = P.getname() + " has no properties left!" ;
        mortage2_message = P.getname() + " is bankrupt!";
        if (!mortage_done)
        {
            P.bankrupt = true;
            mortage_done = true;
        }
        return;
    }
    property* ptr = dynamic_cast<property*>(board[mortage_index]);
    if (ptr != NULL && ptr->owner != NULL && *ptr->owner == P)
    {
        mortage1_message =  "Mortgage " + ptr->getname() + "?(Y/N)";
        mortage2_message = "Value:" + to_string(ptr->getmortage()) + ", lv:" + to_string(ptr->lv);
        mortage_print = true;
        if (yes)
        {
            //resets the property, and give the player money
            P.money += ptr->getmortage();
            ptr->owner = NULL;
            ptr->lv = 1;
            yes = false;
            mortage_index++;//moves on to the next property to ask
            if (mortage_index == 40) mortage_index = 1;
            mortage_print = false;
        }
        else if (no)
        {
            no = false;
            mortage_index++;//moves on to the next property to ask
            if (mortage_index == 40) mortage_index = 1;
            mortage_print = false;
        }
    }
    else
    {
        mortage_index++;
        if (mortage_index == 40) mortage_index = 1;
    }
}
void turn(player &P)
{
    if (P.dead)//player has already been eliminated
    {
        turn_complete = true;
        return;
    }
    if (P.skip != 0)//player skips this turn
    {
        turn_message = "Turn:" + P.getname();
        money_message = "Balance:" + to_string(P.money) + "$";

        game1_message =  P.getname() + " skips this turn.";

        over_message = "Press 'A' to continue......";
        if (enter)
        {
            P.skip --;
            turn_complete = true;
        }
        return;
    }
    if (P.bankrupt)//player is bankrupt, will eliminate player after turn is over
    {
        turn_message = "Turn:" + P.getname();
        money_message = "Balance:" + to_string(P.money) + "$";
        dice_message = "You rolled " + to_string (dice);
        game1_message =  "Pos:" + board[P.pos]->getname();
        if (dynamic_cast<property*>(board[P.pos]) != NULL)
        {
            game2_message = "You paid " + to_string(dynamic_cast<property*>(board[P.pos])->getrent()) +"$ to " + dynamic_cast<property*>(board[P.pos])->owner->getname() +".";
        }
        mortage1_message = P.getname() + " has no properties left!" ;
        mortage2_message = P.getname() + " is bankrupt!";
        over_message = "Press 'A' to continue......";
        if (enter)
        {
            turn_complete = true;
            player::bankrupt_count++;
            P.dead = true;
        }
        return;
    }
    if (rolled)
    {
        roll_dice(P);
        board[after_roll_pos]->block_function(P);
        if (blk_run)
        {
            mortage_check(P);
            if (mortage_done)
            {
                over_message = "Press 'A' to continue......";
                if (enter)
                {
                    turn_complete = true;
                    P.skip += skips;
                    if (paid_flag)
                    {
                        P.paid_student = true;
                        paid_flag = false;
                    }
                    skips = 0;
                    dice = 0;
                }
            }

        }
    }
    turn_message = ("Turn:" + P.getname());
    money_message = ("Balance:" + to_string(P.money) + "$");
    display(P);

}
SDL_Texture* loadTexture( std::string path )
{
    //The final texture
    SDL_Texture* newTexture = NULL;

    //Load image at specified path
    SDL_Surface* loadedSurface = IMG_Load( path.c_str() );
    if( loadedSurface == NULL )
    {
        printf( "Unable to load image %s! SDL_image Error: %s\n", path.c_str(), IMG_GetError() );
    }
    else
    {
        //Create texture from surface pixels
        newTexture = SDL_CreateTextureFromSurface( gRenderer, loadedSurface );
        if( newTexture == NULL )
        {
            printf( "Unable to create texture from %s! SDL Error: %s\n", path.c_str(), SDL_GetError() );
        }

        //Get rid of old loaded surface
        SDL_FreeSurface( loadedSurface );
    }
    return newTexture;
}
