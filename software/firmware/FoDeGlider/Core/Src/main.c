/* USER CODE BEGIN Header */
/**
  ******************************************************************************
  * @file           : main.c
  * @brief          : Main program body
  ******************************************************************************
  * @attention
  *
  * Copyright (c) 2025 STMicroelectronics.
  * All rights reserved.
  *
  * This software is licensed under terms that can be found in the LICENSE file
  * in the root directory of this software component.
  * If no LICENSE file comes with this software, it is provided AS-IS.
  *
  ******************************************************************************
  */

/* USER CODE END Header */
/* Includes ------------------------------------------------------------------*/
#include "main.h"
#include "dma.h"
#include "tim.h"
#include "usart.h"
#include "gpio.h"

/* Private includes ----------------------------------------------------------*/
/* USER CODE BEGIN Includes */

#include "string.h"
#include "stdio.h"
#include "Emm_V5.h"
#include "hipnuc_dec.h"
#include <stdarg.h>

/* USER CODE END Includes */

/* Private typedef -----------------------------------------------------------*/
/* USER CODE BEGIN PTD */

/* USER CODE END PTD */

/* Private define ------------------------------------------------------------*/
/* USER CODE BEGIN PD */

#define RXBUFFERSIZE  256
// sw1210sg+
// 600 0 deg 
// 1490 90 deg
// 2380 180 deg
uint16_t cnt_l = 2380;
uint16_t cnt_r = 600;
float deg_l = 10;
float deg_r = 10;
// AOA NANGU
// NANGU 2535-455
// 455 0 deg
// 2535 180deg
// 455 -90 deg
// 1500 0 deg
// 2535 90 deg
float aoa_deg_l = 0;
float aoa_deg_r = 0;

uint16_t cnt_b = 1500;
uint16_t cnt_b_bias = 5;

uint8_t cmd[16] = {0};
float reduction_ratio = 3591.0f / 187.0f;
float deg2pulse = 160.0f/18.0f * 3591.0f/187.0f;
float Dpulse = 160.0f/18.0f;
float ml2pulse = 60.0f / 2.0f /100.0f * 360.0f * 160.0f/18.0f * 3591.0f/187.0f;
float mm2pulse = 360.0f/ 4.0f * 160.0f/18.0f;

float deg_id_1 = 0;
float mm_id_2 = 0;
float ml_id_3 = 0;

/* Enable/Disable fixed data array decoding example */
#define ENABLE_FIXED_DATA_EXAMPLE     0
/* Enable/Disable DMA for USART reception */
#define ENABLE_USART_DMA              0

#define UART_RX_BUF_SIZE        (1024)
#define LOG_STRING_SIZE         (1024)
/* IMU stream read/control struct */
static hipnuc_raw_t hipnuc_raw = {0};
/* 0: no new data arrived, 1: new data arrived */
uint8_t new_imu_data_flag = 0;
/* The char buffer used to show result */
static char log_buf[LOG_STRING_SIZE];
uint8_t uart_rx_buf[UART_RX_BUF_SIZE];
uint16_t uart_rx_index = 0;

uint8_t message_get = 0;
uint8_t parse_ctrl_get = 0;
/* USER CODE END PD */

/* Private macro -------------------------------------------------------------*/
/* USER CODE BEGIN PM */

/* USER CODE END PM */

/* Private variables ---------------------------------------------------------*/

/* USER CODE BEGIN PV */
uint8_t buffer[1];
char RxBuffer[RXBUFFERSIZE];

float pos = 0.0f, Motor_Cur_Pos = 0.0f, Motor1_Cur_Pos = 0.0f, Motor2_Cur_Pos = 0.0f, Motor3_Cur_Pos = 0.0f;

/* USER CODE END PV */

/* Private function prototypes -----------------------------------------------*/
void SystemClock_Config(void);
/* USER CODE BEGIN PFP */

uint16_t deg2PWM_l(float deg_l)
{
	float deg0 = 180 + deg_l - 360/25;
	return (uint16_t)((deg0)/180.0*(2450-450)+470);
}

uint16_t deg2PWM_r(float deg_r)
{
	// 25T
	// 360/25
	return (uint16_t)((deg_r + 360/25 )/180.0*(2450-450)+450);
}

uint16_t deg2PWM_aoa_l(float aoa_deg_l)
{

	return (uint16_t)((aoa_deg_l)/170.0*(-2550+500) + 2550-960);
}

uint16_t deg2PWM_aoa_r(float aoa_deg_r)
{
	return (uint16_t)((aoa_deg_r)/170.0*(2500-550)+650+700); 
}

void uart2_send_str(const char *s) {
	HAL_UART_Transmit(&huart2, (uint8_t*)s, (uint16_t)strlen(s), 1000);
}

void uart2_send_bytes(const uint8_t *data, uint16_t len) {
	HAL_UART_Transmit(&huart2, (uint8_t*)data, len, 1000);
}

void uart2_printf(const char *fmt, ...) {
	char buf[128];
	va_list ap;
	va_start(ap, fmt);
	int n = vsnprintf(buf, sizeof(buf), fmt, ap);
	va_end(ap);
	if (n > 0) {
		if (n > (int)sizeof(buf)) n = (int)sizeof(buf);
		HAL_UART_Transmit(&huart2, (uint8_t*)buf, (uint16_t)n, 1000);
	}
}

void print_message(){
	// stepper1
	printf("step1");
	Emm_V5_Read_Sys_Params(1,S_CPOS);
	while(rxFrameFlag == false); rxFrameFlag = false; 
	HAL_Delay(1000);
	if(rxCmd[0] == 1 && rxCmd[1] == 0x36 && rxCount == 8)
	{
		pos = (uint32_t)(
											((uint32_t)rxCmd[3] << 24)    |
											((uint32_t)rxCmd[4] << 16)    |
											((uint32_t)rxCmd[5] << 8)     |
											((uint32_t)rxCmd[6] << 0)
										);

		Motor1_Cur_Pos = (float)pos * 360.0f / 65536.0f;

		if(rxCmd[2]) { Motor1_Cur_Pos = -Motor1_Cur_Pos; }
	}
	printf("Stepper1, Spin Cur_Pos = %.3f deg \r\n", (Motor1_Cur_Pos*Dpulse/deg2pulse));
	// stepper2
	Emm_V5_Read_Sys_Params(2,S_CPOS);
	while(rxFrameFlag == false); rxFrameFlag = false; 
	HAL_Delay(1000);
	if(rxCmd[0] == 2 && rxCmd[1] == 0x36 && rxCount == 8)
	{
		pos = (uint32_t)(
											((uint32_t)rxCmd[3] << 24)    |
											((uint32_t)rxCmd[4] << 16)    |
											((uint32_t)rxCmd[5] << 8)     |
											((uint32_t)rxCmd[6] << 0)
										);

		Motor2_Cur_Pos = (float)pos * 360.0f / 65536.0f;

		if(rxCmd[2]) { Motor2_Cur_Pos = -Motor2_Cur_Pos; }
	}
	printf("Stepper2, Trans Cur_Pos = %.3f mm\r\n", (Motor2_Cur_Pos*Dpulse/mm2pulse));
	// stepper3
	Emm_V5_Read_Sys_Params(3,S_CPOS);
	while(rxFrameFlag == false); rxFrameFlag = false; 
	HAL_Delay(1000);
	if(rxCmd[0] == 3 && rxCmd[1] == 0x36 && rxCount == 8)
	{
		pos = (uint32_t)(
											((uint32_t)rxCmd[3] << 24)    |
											((uint32_t)rxCmd[4] << 16)    |
											((uint32_t)rxCmd[5] << 8)     |
											((uint32_t)rxCmd[6] << 0)
										);

		Motor3_Cur_Pos = (float)pos * 360.0f / 65536.0f;

		if(rxCmd[2]) { Motor3_Cur_Pos = -Motor3_Cur_Pos; }
	}
	printf("Stepper3, water Cur_Pos = %.3f ml\r\n", (Motor3_Cur_Pos*Dpulse/ml2pulse));
	uart2_printf("Stepper1, Spin Cur_Pos = %.3f deg. Stepper2, Trans Cur_Pos = %.3f mm. Stepper 3, water Cur_Pos = %.3f ml\r\n", (Motor1_Cur_Pos*Dpulse/deg2pulse), (Motor2_Cur_Pos*Dpulse/mm2pulse), (Motor3_Cur_Pos*Dpulse/ml2pulse));
}
void parse_ctrl(uint8_t buffer[])
{
	if((char)buffer[0] == 'a')
	{
		printf("Order: %c! Left wing open! ", buffer[0]);
		if(deg_l + 5 > 0)
		{
			deg_l = 0;
		}else{
			deg_l = deg_l + 5; 
		}
		uart2_printf("STM32: Open left wing! left deg = %.1f\r\n", deg_l);		
		printf("The deg_l is %.1f!\r\n", deg_l);
		__HAL_TIM_SetCompare(&htim4,TIM_CHANNEL_1,deg2PWM_l(deg_l)-1);
	}
	if((char)buffer[0] == 'z')
	{
		printf("Order: %c! Left wing close! ", buffer[0]);
		if(deg_l - 5 < -90)
		{
			deg_l = -90;
		}else{
			deg_l = deg_l - 5; 
		}
		uart2_printf("STM32: Close left wing! left deg = %.1f\r\n", deg_l);
		printf("The deg_l is %.1f!\n", deg_l);
		__HAL_TIM_SetCompare(&htim4,TIM_CHANNEL_1,deg2PWM_l(deg_l)-1);
	}
	
	if((char)buffer[0] == 'd')
	{
		printf("Order: %c! Right wing open! ", buffer[0]);
		if(deg_r - 5 < 0)
		{
			deg_r = 0;
		}else{
			deg_r = deg_r - 5; 
		}
		uart2_printf("STM32: Open Right wing! Right deg = %.1f\r\n", deg_r);
		printf("The deg_r is %.1f!\n", deg_r);
		__HAL_TIM_SetCompare(&htim4,TIM_CHANNEL_2,deg2PWM_r(deg_r)-1);
	}
	if((char)buffer[0] == 'c')
	{
		printf("Order: %c! right wing close! ", buffer[0]);
		if(deg_r + 5 > 90)
		{
			deg_r = 90;
		}else{
			deg_r = deg_r + 5; 
		}
		uart2_printf("STM32: Close Right wing! Right deg = %.1f\r\n", deg_r);
		printf("The deg_r is %.1f!\n", deg_r);
		__HAL_TIM_SetCompare(&htim4,TIM_CHANNEL_2,deg2PWM_r(deg_r)-1);
	}
	
	if((char)buffer[0] == 's')
	{
		printf("Order: %c! Speed up! ", buffer[0]);
		cnt_b = 1565;
		uart2_printf("STM32: Speed up! speed pwm = %u\r\n", cnt_b);
		printf("The speed is %u! The 0 speed is 1500. \n", cnt_b);
		__HAL_TIM_SetCompare(&htim4,TIM_CHANNEL_3,cnt_b-1);
	}
	if((char)buffer[0] == 'x')
	{
		printf("Order: %c! Slow down! ", buffer[0]);
		cnt_b = 1500;
		uart2_printf("STM32: Speed down! speed pwm = %u\r\n", cnt_b);
		printf("The speed is %u! The 0 speed is 1500. \n", cnt_b);
		__HAL_TIM_SetCompare(&htim4,TIM_CHANNEL_3,cnt_b-1);
	}
	
	if((char)buffer[0] == 'r') // ready
	{
		printf("Order: %c! Ready!", buffer[0]);
		HAL_GPIO_WritePin(GPIOF,GPIO_PIN_2,GPIO_PIN_SET);

		deg_l = 90;
		__HAL_TIM_SetCompare(&htim4,TIM_CHANNEL_1,deg2PWM_l(deg_l)-1);
		// propeller
		__HAL_TIM_SetCompare(&htim4,TIM_CHANNEL_3,cnt_b-1);
		aoa_deg_l = 0;
		aoa_deg_r = 0;
		__HAL_TIM_SetCompare(&htim3,TIM_CHANNEL_1,deg2PWM_aoa_l(aoa_deg_l)-1);
		__HAL_TIM_SetCompare(&htim3,TIM_CHANNEL_2,deg2PWM_aoa_r(aoa_deg_r)-1);
	}	
	if((char)buffer[0] == 'p'){
		printf("step0");
		print_message();
	}
	
	if((char)buffer[0] == 'j')
	{
		printf("Order: %c! Left row!", buffer[0]);
		deg_id_1 = deg_id_1 + 1;
		if (deg_id_1 < 0){
			if (deg_id_1 < -25){
				deg_id_1 = -25;
			}
			Emm_V5_Pos_Control(1, 0, 80, 0, (uint32_t)( - deg_id_1 * deg2pulse), 1, 0);
			}
		else{
			if (deg_id_1 > 25){
				deg_id_1 = 25;
			}
			Emm_V5_Pos_Control(1, 1, 80, 0, (uint32_t)(deg_id_1 * deg2pulse), 1, 0);
			}
		uart2_printf("STM32: Left row! Now spin = %.1f deg\r\n", deg_id_1);
	}	
	if((char)buffer[0] == 'l')
	{
		printf("Order: %c! Right row!", buffer[0]);
		deg_id_1 = deg_id_1 - 1;
		if (deg_id_1 < 0){
			if (deg_id_1 < -25){
				deg_id_1 = -25;
			}
			Emm_V5_Pos_Control(1, 0, 80, 0, (uint32_t)( - deg_id_1 * deg2pulse), 1, 0);
			}
		else{
			if (deg_id_1 > 25){
				deg_id_1 = 25;
			}
			Emm_V5_Pos_Control(1, 1, 80, 0, (uint32_t)(deg_id_1 * deg2pulse), 1, 0);
			}
		uart2_printf("STM32: Right row! Now spin = %.1f deg\r\n", deg_id_1);	
	}	
	
	if((char)buffer[0] == 'i') // pitch
	{
		printf("Order: %c! High pitch!", buffer[0]);
		mm_id_2 = mm_id_2 + 5;
		if (mm_id_2 < 0){ 
			if (mm_id_2 < -55){
				mm_id_2 = -55;
			}
			Emm_V5_Pos_Control(2, 1, 100, 0, (uint32_t)(- mm_id_2 * mm2pulse), 1, 0); 
		}
		else{
			if (mm_id_2 > 55){
				mm_id_2 = 55;
			}
			Emm_V5_Pos_Control(2, 0, 100, 0, (uint32_t)( mm_id_2 * mm2pulse), 1, 0);
		}
		uart2_printf("STM32: High pitch! Trans pos = %.1f mm\r\n", mm_id_2);
	}	
	if((char)buffer[0] == 'k') // pitch
	{
		printf("Order: %c! Low pitch!", buffer[0]);
		mm_id_2 = mm_id_2 - 5;
		if (mm_id_2 < 0){ 
			if (mm_id_2 < -55){
				mm_id_2 = -55;
			}
			Emm_V5_Pos_Control(2, 1, 100, 0, (uint32_t)(- mm_id_2 * mm2pulse), 1, 0); 
		}
		else{
			if (mm_id_2 > 55){
				mm_id_2 = 55;
			}
			Emm_V5_Pos_Control(2, 0, 100, 0, (uint32_t)( mm_id_2 * mm2pulse), 1, 0);
		}
		uart2_printf("STM32: Low pitch! Trans pos = %.1f mm\r\n", mm_id_2);
	}	
	
	if((char)buffer[0] == 'y')
	{
		printf("Order: %c! drink water!", buffer[0]);
		ml_id_3 = ml_id_3 + 1;
		if (ml_id_3 < 0){
			ml_id_3 = 0;
			Emm_V5_Pos_Control(3, 1, 500, 0, (uint32_t)(ml_id_3 * ml2pulse), 1, 0);
		}
		else{
			if (ml_id_3 > 100){
				ml_id_3 = 100;
			}
		Emm_V5_Pos_Control(3, 1, 500, 0, (uint32_t)( ml_id_3 * ml2pulse), 1, 0);
		}
		uart2_printf("STM32: Drink! water = %.1f\r\n", ml_id_3);
	}	
	
	if((char)buffer[0] == 'h')
	{
		printf("Order: %c! drainage!", buffer[0]);
		ml_id_3 = ml_id_3 - 1;
		if (ml_id_3 < 0){
			ml_id_3 = 0;
			Emm_V5_Pos_Control(3, 1, 500, 0, (uint32_t)(ml_id_3 * ml2pulse), 1, 0);
		}
		else{
			if (ml_id_3 > 100){
				ml_id_3 = 100;
			}
		Emm_V5_Pos_Control(3, 1, 500, 0, (uint32_t)( ml_id_3 * ml2pulse), 1, 0);
		}
		uart2_printf("STM32: Drainage! water = %.1f\r\n", ml_id_3);
	}
	
// AOA
	if((char)buffer[0] == 'f')
	{
		printf("Order: %c! Left wing bigger AOA! ", buffer[0]);
		if(aoa_deg_l + 5 > 90)
		{
			aoa_deg_l = 90;
		}else{
			aoa_deg_l = aoa_deg_l + 5; 
		}
		printf("The aoa_deg_l is %.1f!\n", aoa_deg_l);
		__HAL_TIM_SetCompare(&htim3,TIM_CHANNEL_1,deg2PWM_aoa_l(aoa_deg_l)-1);
		uart2_printf("STM32: Bigger left AOA! aoa_deg_l = %.1f\r\n", aoa_deg_l);
	}
	if((char)buffer[0] == 'v')
	{
		printf("Order: %c! Left wing smaller AOA! ", buffer[0]);
		if(aoa_deg_l - 5 < -80)
		{
			aoa_deg_l = -80;
		}else{
			aoa_deg_l = aoa_deg_l - 5; 
		}
		printf("The aoa_deg_l is %.1f!\n", aoa_deg_l);
		__HAL_TIM_SetCompare(&htim3,TIM_CHANNEL_1,deg2PWM_aoa_l(aoa_deg_l)-1);
		uart2_printf("STM32: Smaller left AOA! aoa_deg_l = %.1f\r\n", aoa_deg_l);
	}
	
	if((char)buffer[0] == 'g')
	{
		printf("Order: %c! Right wing bigger AOA! ", buffer[0]);
		if(aoa_deg_r + 5 > 90)
		{
			aoa_deg_r = 90;
		}else{
			aoa_deg_r = aoa_deg_r + 5; 
		}
		printf("The aoa_deg_r is %.1f!\n", aoa_deg_r);
		__HAL_TIM_SetCompare(&htim3,TIM_CHANNEL_2,deg2PWM_aoa_r(aoa_deg_r)-1);
		uart2_printf("STM32: Bigger Right AOA! aoa_deg_r = %.1f\r\n", aoa_deg_r);
	}
	if((char)buffer[0] == 'b')
	{
		printf("Order: %c! right wing smaller AOA! ", buffer[0]);
		if(aoa_deg_r - 5 < -80)
		{
			aoa_deg_r = -80;
		}else{
			aoa_deg_r = aoa_deg_r - 5; 
		}
		printf("The aoa_deg_r is %.1f!\n", aoa_deg_r);
		__HAL_TIM_SetCompare(&htim3,TIM_CHANNEL_2,deg2PWM_aoa_r(aoa_deg_r)-1);
		uart2_printf("STM32: Smaller Right AOA! aoa_deg_r = %.1f\r\n", aoa_deg_r);
	}
	
}

int fputc(int ch, FILE *f)
{
  HAL_UART_Transmit(&huart1, (uint8_t *)&ch, 1, 0xffff);
  return ch;
}
 
int fgetc(FILE *f)
{
  uint8_t ch = 0;
  HAL_UART_Receive(&huart1, (uint8_t *)&ch, 1, 0xffff);
  return ch;
}

static void process_data(void);

/* USER CODE END PFP */

/* Private user code ---------------------------------------------------------*/
/* USER CODE BEGIN 0 */

/* USER CODE END 0 */

/**
  * @brief  The application entry point.
  * @retval int
  */
int main(void)
{

  /* USER CODE BEGIN 1 */

  /* USER CODE END 1 */

  /* Reset of all peripherals, Initializes the Flash interface and the Systick. */
  HAL_Init();

  /* USER CODE BEGIN Init */
	
	/* Initialize instance */
	memset(&hipnuc_raw, 0, sizeof(hipnuc_raw_t));
	new_imu_data_flag = 0;
	/* Use macro to control whether to show the fixed data array decoding example */
	#if ENABLE_FIXED_DATA_EXAMPLE
	#include "example_data.h"
	process_example_data();
	#endif
	
  /* USER CODE END Init */

  /* Configure the system clock */
  SystemClock_Config();

  /* USER CODE BEGIN SysInit */

  /* USER CODE END SysInit */

  /* Initialize all configured peripherals */
  MX_GPIO_Init();
  MX_DMA_Init();
  MX_TIM4_Init();
  MX_USART2_UART_Init();
  MX_USART1_UART_Init();
  MX_USART3_UART_Init();
  MX_USART6_UART_Init();
  MX_TIM3_Init();
  /* USER CODE BEGIN 2 */
	HAL_Delay(1000);
	HAL_TIM_PWM_Start(&htim4,TIM_CHANNEL_1);
	HAL_TIM_PWM_Start(&htim4,TIM_CHANNEL_2);
	HAL_TIM_PWM_Start(&htim4,TIM_CHANNEL_3);
	// AOA
	HAL_TIM_PWM_Start(&htim3,TIM_CHANNEL_1);
	HAL_TIM_PWM_Start(&htim3,TIM_CHANNEL_2);
	
	// DMA
	HAL_UART_Receive_DMA(&huart2,buffer,1);
	__HAL_UART_ENABLE_IT(&huart2,UART_IT_IDLE);
	__HAL_UART_ENABLE_IT(&huart3, UART_IT_IDLE);
	HAL_UART_Receive_DMA(&huart3, (uint8_t *)rxCmd, CMD_LEN);

	__HAL_UART_ENABLE_IT(&huart6, UART_IT_RXNE);
	__HAL_UART_ENABLE_IT(&huart6, UART_IT_IDLE);
	
	printf("Nihao");
	
  /* USER CODE END 2 */

  /* Infinite loop */
  /* USER CODE BEGIN WHILE */
	
	cnt_b = 1500;
	__HAL_TIM_SetCompare(&htim4,TIM_CHANNEL_3,cnt_b-1);
	HAL_Delay(5000);
	HAL_GPIO_WritePin(GPIOF,GPIO_PIN_2,GPIO_PIN_SET);
	HAL_Delay(1000);
	deg_l = 0;
	deg_r = 0;
	__HAL_TIM_SetCompare(&htim4,TIM_CHANNEL_1,deg2PWM_l(deg_l));
	__HAL_TIM_SetCompare(&htim4,TIM_CHANNEL_2,deg2PWM_r(deg_r));
	aoa_deg_l = 0;
	aoa_deg_r = 0;
	__HAL_TIM_SetCompare(&htim3,TIM_CHANNEL_1,deg2PWM_aoa_l(aoa_deg_l)-1);
	__HAL_TIM_SetCompare(&htim3,TIM_CHANNEL_2,deg2PWM_aoa_r(aoa_deg_r)-1);
	HAL_Delay(500);
  printf("Begin! \r\n");
	while (1)
  {
		if(parse_ctrl_get){
			parse_ctrl(buffer);
			parse_ctrl_get = 0;
		}
		if(new_imu_data_flag){
			process_data();
		}
		
    /* USER CODE END WHILE */

    /* USER CODE BEGIN 3 */
  }
  /* USER CODE END 3 */
}

/**
  * @brief System Clock Configuration
  * @retval None
  */
void SystemClock_Config(void)
{
  RCC_OscInitTypeDef RCC_OscInitStruct = {0};
  RCC_ClkInitTypeDef RCC_ClkInitStruct = {0};

  /** Configure the main internal regulator output voltage
  */
  __HAL_RCC_PWR_CLK_ENABLE();
  __HAL_PWR_VOLTAGESCALING_CONFIG(PWR_REGULATOR_VOLTAGE_SCALE1);

  /** Initializes the RCC Oscillators according to the specified parameters
  * in the RCC_OscInitTypeDef structure.
  */
  RCC_OscInitStruct.OscillatorType = RCC_OSCILLATORTYPE_HSI;
  RCC_OscInitStruct.HSIState = RCC_HSI_ON;
  RCC_OscInitStruct.HSICalibrationValue = RCC_HSICALIBRATION_DEFAULT;
  RCC_OscInitStruct.PLL.PLLState = RCC_PLL_ON;
  RCC_OscInitStruct.PLL.PLLSource = RCC_PLLSOURCE_HSI;
  RCC_OscInitStruct.PLL.PLLM = 8;
  RCC_OscInitStruct.PLL.PLLN = 168;
  RCC_OscInitStruct.PLL.PLLP = RCC_PLLP_DIV2;
  RCC_OscInitStruct.PLL.PLLQ = 4;
  if (HAL_RCC_OscConfig(&RCC_OscInitStruct) != HAL_OK)
  {
    Error_Handler();
  }

  /** Initializes the CPU, AHB and APB buses clocks
  */
  RCC_ClkInitStruct.ClockType = RCC_CLOCKTYPE_HCLK|RCC_CLOCKTYPE_SYSCLK
                              |RCC_CLOCKTYPE_PCLK1|RCC_CLOCKTYPE_PCLK2;
  RCC_ClkInitStruct.SYSCLKSource = RCC_SYSCLKSOURCE_PLLCLK;
  RCC_ClkInitStruct.AHBCLKDivider = RCC_SYSCLK_DIV1;
  RCC_ClkInitStruct.APB1CLKDivider = RCC_HCLK_DIV4;
  RCC_ClkInitStruct.APB2CLKDivider = RCC_HCLK_DIV2;

  if (HAL_RCC_ClockConfig(&RCC_ClkInitStruct, FLASH_LATENCY_5) != HAL_OK)
  {
    Error_Handler();
  }
}

/* USER CODE BEGIN 4 */

/**
 * @brief Process IMU data
 * 
 * If new data is available, processes the received IMU data by decoding it and printing the results.
 */
static void process_data(void)
{
    if (new_imu_data_flag)
    {
        new_imu_data_flag = 0;

        for (uint16_t i = 0; i < uart_rx_index; i++)
        {
            if (hipnuc_input(&hipnuc_raw, uart_rx_buf[i]))
            {
                /* Convert result to strings */
                hipnuc_dump_packet(&hipnuc_raw, log_buf, sizeof(log_buf));
                /* Display result */
							printf("roll: %.3f, pitch: %.3f\r\n", hipnuc_raw.hi91.roll, hipnuc_raw.hi91.pitch);
            }
        }

        uart_rx_index = 0; // Reset buffer index after processing
    }
}
/* USER CODE END 4 */

/**
  * @brief  This function is executed in case of error occurrence.
  * @retval None
  */
void Error_Handler(void)
{
  /* USER CODE BEGIN Error_Handler_Debug */
  /* User can add his own implementation to report the HAL error return state */
  __disable_irq();
  while (1)
  {
  }
  /* USER CODE END Error_Handler_Debug */
}

#ifdef  USE_FULL_ASSERT
/**
  * @brief  Reports the name of the source file and the source line number
  *         where the assert_param error has occurred.
  * @param  file: pointer to the source file name
  * @param  line: assert_param error line source number
  * @retval None
  */
void assert_failed(uint8_t *file, uint32_t line)
{
  /* USER CODE BEGIN 6 */
  /* User can add his own implementation to report the file name and line number,
     ex: printf("Wrong parameters value: file %s on line %d\r\n", file, line) */
  /* USER CODE END 6 */
}
#endif /* USE_FULL_ASSERT */
